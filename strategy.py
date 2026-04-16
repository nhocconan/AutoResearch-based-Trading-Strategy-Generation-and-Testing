#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h/1d regime filters
# Long when: 1h RSI < 30 AND price > 4h VWAP AND 1d close > 1d 200 EMA (bullish bias)
# Short when: 1h RSI > 70 AND price < 4h VWAP AND 1d close < 1d 200 EMA (bearish bias)
# Exit when 1h RSI crosses 50 (mean reversion complete) or ATR stoploss (1.5*ATR)
# Uses discrete size 0.20. Target: 80-120 total trades over 4 years (20-30/year).
# Works in bull markets via 1d EMA200 filter and in bear markets via short side.
# Session filter 08-20 UTC reduces noise and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators ===
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # VWAP approximation using typical price
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).cumsum()
    vwap_den = pd.Series(volume).cumsum()
    vwap = (vwap_num / vwap_den).values
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 4h Indicators: VWAP (same calculation as 1h but on 4h data) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    vwap_num_4h = pd.Series(typical_price_4h * volume_4h).cumsum()
    vwap_den_4h = pd.Series(volume_4h).cumsum()
    vwap_4h = (vwap_num_4h / vwap_den_4h).values
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # === 1d Indicators: EMA(200) for trend bias ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 200
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi_values[i]) or np.isnan(vwap[i]) or np.isnan(vwap_4h_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_1h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        rsi_val = rsi_values[i]
        price_vwap = vwap[i]
        vwap_4h = vwap_4h_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        atr_val = atr_1h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI crosses above 50 (mean reversion complete)
            if rsi_val > 50:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI crosses below 50 (mean reversion complete)
            if rsi_val < 50:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI < 30 (oversold) AND price > 4h VWAP (4h bullish bias) AND 1d close > 1d EMA200 (bullish trend)
            if rsi_val < 30 and price > vwap_4h and close_1d[-1] > ema_200:  # close_1d[-1] is latest 1d close
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: RSI > 70 (overbought) AND price < 4h VWAP (4h bearish bias) AND 1d close < 1d EMA200 (bearish trend)
            elif rsi_val > 70 and price < vwap_4h and close_1d[-1] < ema_200:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSIMeanRev_4hVWAP_1dEMA200_V1"
timeframe = "1h"
leverage = 1.0