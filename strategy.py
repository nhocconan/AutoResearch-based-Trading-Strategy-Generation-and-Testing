#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 AND 1d close > 1d open (bullish daily candle) AND volume > 1.5x 20-period median.
# Short when price breaks below S3 AND 1d close < 1d open (bearish daily candle) AND volume > 1.5x 20-period median.
# Uses ATR-based stoploss: exit long if price < highest_high_since_entry - 2.0*ATR(14),
# exit short if price > lowest_low_since_entry + 2.0*ATR(14).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 12h timeframe.
# Camarilla levels provide institutional support/resistance; 1d candle direction filters false breakouts.

name = "12h_Camarilla_R3S3_Breakout_1dCandleDir_VolumeSpike_ATR_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Calculate 1d trend filter: bullish/bearish daily candle (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 1d candle direction: 1 = bullish (close > open), -1 = bearish (close < open), 0 = doji
    daily_bullish = (df_1d['close'] > df_1d['open']).astype(int)
    daily_bearish = (df_1d['close'] < df_1d['open']).astype(int)
    daily_trend = daily_bullish - daily_bearish  # 1 for bullish, -1 for bearish, 0 for doji
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend.values)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low),
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 for breakout entries
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    prev_1d_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_1d_high - prev_1d_low
    r3 = prev_1d_close + 1.1 * camarilla_range
    s3 = prev_1d_close - 1.1 * camarilla_range
    
    # Al Camarilla levels to 12h timeframe (wait for 1d close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for ATR, volume, and aligned indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(daily_trend_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND 1d bullish candle AND volume spike
            if curr_close > r3_aligned[i] and daily_trend_aligned[i] > 0 and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Price breaks below S3 AND 1d bearish candle AND volume spike
            elif curr_close < s3_aligned[i] and daily_trend_aligned[i] < 0 and volume_confirm:
                signals[i] = -0.25
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
            
            # Exit conditions: ATR stoploss OR Camarilla S3 break OR trend reversal
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or curr_close < s3_aligned[i] or daily_trend_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Camarilla R3 break OR trend reversal
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or curr_close > r3_aligned[i] or daily_trend_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals