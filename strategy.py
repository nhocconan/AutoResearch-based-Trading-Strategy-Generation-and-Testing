#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w/1d HTF trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1w EMA34 AND volume > 1.8x 24-bar avg
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1w EMA34 AND volume > 1.8x 24-bar avg
# Exit when Alligator alignment reverses or volume drops below 1.2x average
# Uses Williams Alligator (SMAs with specific offsets) to identify trend phases
# 1w EMA34 filter ensures we only trade with the long-term trend
# Volume confirmation ensures breakouts have conviction
# Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)

name = "12h_WilliamsAlligator_1wEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator components on 12h data (using median prices)
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars ahead
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars ahead
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMMA, shifted 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars ahead
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Volume confirmation: >1.8x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    volume_exit = volume < 1.2 * volume_ma_24  # exit when volume drops
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 13, 8, 5)  # Volume MA(24) and Alligator components
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        vol_exit_cond = volume_exit[i]
        curr_close = close[i]
        ema_34 = ema_34_1w_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Check Alligator alignment
        bullish_alignment = jaw_val < teeth_val < lips_val  # jaws < teeth < lips
        bearish_alignment = jaw_val > teeth_val > lips_val  # jaws > teeth > lips
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when bullish alignment AND price > 1w EMA34 AND volume confirmation
            if bullish_alignment and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND price < 1w EMA34 AND volume confirmation
            elif bearish_alignment and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when alignment reverses or volume drops
            if not bullish_alignment or vol_exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when alignment reverses or volume drops
            if not bearish_alignment or vol_exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals