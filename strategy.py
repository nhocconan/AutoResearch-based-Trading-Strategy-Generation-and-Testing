#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivots provide intraday/reversal levels that work on daily timeframe.
# 1w EMA34 ensures alignment with weekly trend. Volume spike (>2.0x 20-day average) filters chop.
# Discrete position sizing (0.25) minimizes fee churn. Target 7-25 trades/year to avoid overtrading.
# Works in bull/bear markets by fading false breaks in range and capturing real breakouts in trend.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivots from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    
    # Camarilla levels (based on previous bar)
    R3 = close + (range_hl * 1.1 / 4)
    S3 = close - (range_hl * 1.1 / 4)
    R4 = close + (range_hl * 1.1 / 2)
    S4 = close - (range_hl * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 1  # Need previous bar for pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(typical_price[i-1]) or np.isnan(range_hl[i-1])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: >2.0x 20-day average volume
        if i >= 20:
            volume_ma_20 = np.mean(volume[max(0, i-19):i+1])
            vol_confirm = volume[i] > 2.0 * volume_ma_20
        else:
            vol_confirm = False
        
        price = close[i]
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        # Recalculate Camarilla levels for previous bar
        prev_typical = (prev_high + prev_low + prev_close) / 3.0
        prev_range = prev_high - prev_low
        R3_prev = prev_close + (prev_range * 1.1 / 4)
        S3_prev = prev_close - (prev_range * 1.1 / 4)
        R4_prev = prev_close + (prev_range * 1.1 / 2)
        S4_prev = prev_close - (prev_range * 1.1 / 2)
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Close above R3 with volume spike and price > 1w EMA34
            if price > R3_prev and vol_confirm and price > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Close below S3 with volume spike and price < 1w EMA34
            elif price < S3_prev and vol_confirm and price < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or close below S3
            # ATR-based stoploss: 2.0 * ATR below entry (using 1d ATR)
            if i >= 14:
                tr1 = high[max(0, i-13):i+1] - low[max(0, i-13):i+1]
                tr2 = np.abs(high[max(0, i-13):i+1] - close[max(0, i-13):i])
                tr3 = np.abs(low[max(0, i-13):i+1] - close[max(0, i-13):i])
                tr = np.maximum(np.maximum(tr1, tr2), tr3)
                atr_val = np.mean(tr[-14:])
                stop_loss = entry_price - 2.0 * atr_val
            else:
                stop_loss = entry_price - 2.0 * (np.std(close[max(0, i-13):i+1]) if i >= 1 else price * 0.02)
            
            # Exit on stoploss or when price closes below S3 (failed breakout)
            if price < stop_loss or price < S3_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or close above R3
            # ATR-based stoploss: 2.0 * ATR above entry
            if i >= 14:
                tr1 = high[max(0, i-13):i+1] - low[max(0, i-13):i+1]
                tr2 = np.abs(high[max(0, i-13):i+1] - close[max(0, i-13):i])
                tr3 = np.abs(low[max(0, i-13):i+1] - close[max(0, i-13):i])
                tr = np.maximum(np.maximum(tr1, tr2), tr3)
                atr_val = np.mean(tr[-14:])
                stop_loss = entry_price + 2.0 * atr_val
            else:
                stop_loss = entry_price + 2.0 * (np.std(close[max(0, i-13):i+1]) if i >= 1 else price * 0.02)
            
            # Exit on stoploss or when price closes above R3 (failed breakout)
            if price > stop_loss or price > R3_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals