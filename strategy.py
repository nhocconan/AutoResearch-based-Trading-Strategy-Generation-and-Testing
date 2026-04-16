#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and 1d volume confirmation.
# Long when 1h RSI < 30 AND price > 4h EMA20 (uptrend) AND 1d volume > 1.5x 20-period average.
# Short when 1h RSI > 70 AND price < 4h EMA20 (downtrend) AND 1d volume > 1.5x 20-period average.
# Exit when RSI crosses 50 (mean reversion complete) or ATR-based stop (1.5*ATR from entry).
# Uses discrete position size 0.20. Designed for 1h timeframe with strict filters to limit trades.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by requiring volume confirmation and trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # === 4h Indicators: EMA(20) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_1h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        rsi_val = rsi[i]
        trend = ema_4h_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI crosses above 50 (mean reversion)
            if rsi_val > 50:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI crosses below 50 (mean reversion)
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
            # LONG: RSI < 30 AND price > 4h EMA20 (uptrend) AND volume spike
            if rsi_val < 30 and price > trend and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: RSI > 70 AND price < 4h EMA20 (downtrend) AND volume spike
            elif rsi_val > 70 and price < trend and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSI14_4hEMA20_1dVolumeSpike_V1"
timeframe = "1h"
leverage = 1.0