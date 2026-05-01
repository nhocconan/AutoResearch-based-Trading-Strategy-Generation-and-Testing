#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 (4h Camarilla) AND 4h EMA50 uptrend AND volume > 2.0x 20-period median.
# Short when price breaks below S1 (4h Camarilla) AND 4h EMA50 downtrend AND volume > 2.0x 20-period median.
# Uses ATR(14) stoploss: exit long if price < highest_since_entry - 2.0*ATR(14), exit short if price > lowest_since_entry + 2.0*ATR(14).
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-37 trades/year on 1h timeframe.
# Camarilla levels from 4h provide intraday pivot points that work in both bull and bear markets.
# Volume confirmation ensures breakouts have participation, reducing false signals.
# ATR stoploss adapts to volatility while respecting engine semantics (close-based exit).

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_ATR_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter (loaded once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels (R1, S1, R2, S2) - using previous bar's OHLC
    # Camarilla: R2 = C + ((H-L)*1.1/2), R1 = C + ((H-L)*1.1/4), S1 = C - ((H-L)*1.1/4), S2 = C - ((H-L)*1.1/2)
    # where C = (H+L+C)/3 (typical price), but for intraday we use previous bar's values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Typical price for pivot calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 4.0)
    s1 = pivot - (range_hl * 1.1 / 4.0)
    r2 = pivot + (range_hl * 1.1 / 2.0)
    s2 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_4h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_4h, s2)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA, Camarilla, volume, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 4h EMA50 direction
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R1 AND uptrend AND volume spike
            if curr_close > r1_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Price breaks below S1 AND downtrend AND volume spike
            elif curr_close < s1_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR break below S1 (reversal) OR trend reversal
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or curr_close < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR break above R1 (reversal) OR trend reversal
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or curr_close > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals