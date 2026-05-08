#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Choppiness_Reversal_Signal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = (close_1w > ema34_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Choppiness Index (14-period) - chop > 61.8 = range, chop < 38.2 = trend
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(high[1:] - low[1:], np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])  # TR for each period
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_hl = hh - ll
    chop = np.where((sum_tr14 > 0) & (range_hl > 0), 100 * np.log10(sum_tr14 / range_hl) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.3 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: choppy market (range) + RSI oversold + volume spike + weekly uptrend
            long_cond = (chop[i] > 61.8 and rsi[i] < 35 and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: choppy market (range) + RSI overbought + volume spike + weekly downtrend
            short_cond = (chop[i] > 61.8 and rsi[i] > 65 and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or chop breaks below 38.2 (trending market)
            if rsi[i] > 70 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or chop breaks below 38.2 (trending market)
            if rsi[i] < 30 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Choppiness Index identifies ranging markets (chop > 61.8) where mean reversion works.
# In ranging markets, we take RSI extremes with volume confirmation and weekly trend filter.
# Exits when RSI reverses or market starts trending (chop < 38.2).
# Weekly EMA34 ensures we only trade in direction of higher timeframe trend.
# Target: 15-25 trades/year to minimize fee decay while capturing mean reversion in ranges.