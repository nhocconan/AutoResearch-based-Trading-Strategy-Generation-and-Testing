#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI divergence with 1d trend filter and volume confirmation.
# Bullish divergence: price makes lower low, RSI makes higher low -> long
# Bearish divergence: price makes higher high, RSI makes lower high -> short
# Enter only when aligned with 1d EMA200 trend and volume > 1.5x 20-period average
# Exit on opposite divergence or 2*ATR stoploss
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # === 1d Indicators: EMA200 and Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 220
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Track recent extremes for divergence detection
    lookback = 10
    price_low = np.full(n, np.nan)
    price_high = np.full(n, np.nan)
    rsi_low = np.full(n, np.nan)
    rsi_high = np.full(n, np.nan)
    
    for i in range(lookback, n):
        price_low[i] = np.min(low[i-lookback:i+1])
        price_high[i] = np.max(high[i-lookback:i+1])
        rsi_low[i] = np.min(rsi[i-lookback:i+1])
        rsi_high[i] = np.max(rsi[i-lookback:i+1])
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or 
            np.isnan(price_low[i]) or np.isnan(price_high[i]) or
            np.isnan(rsi_low[i]) or np.isnan(rsi_high[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        curr_low = low[i]
        curr_high = high[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit on bearish divergence or stoploss
            if (price_high[i] == curr_high and rsi_high[i] < rsi[i-lookback] and 
                price_low[i-lookback] > price_low[i]):
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit on bullish divergence or stoploss
            if (price_low[i] == curr_low and rsi_low[i] > rsi[i-lookback] and 
                price_high[i-lookback] < price_high[i]):
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # BULLISH DIVERGENCE: price makes lower low, RSI makes higher low
            bull_div = (price_low[i] == curr_low and 
                       price_low[i] < price_low[i-lookback] and
                       rsi_low[i] > rsi_low[i-lookback])
            
            # BEARISH DIVERGENCE: price makes higher high, RSI makes lower high
            bear_div = (price_high[i] == curr_high and 
                       price_high[i] > price_high[i-lookback] and
                       rsi_high[i] < rsi_high[i-lookback])
            
            # LONG: Bullish divergence AND price > 1d EMA200 AND volume spike
            if bull_div and price > ema_200_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bearish divergence AND price < 1d EMA200 AND volume spike
            elif bear_div and price < ema_200_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_RSIDivergence_1dEMA200_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0