#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Camarilla levels (R3/S3, R4/S4) provide institutional breakout/fade zones
# - Breakout at R4/S4 with volume confirmation captures strong momentum moves
# - Fade at R3/S3 with trend filter avoids counter-trend traps in ranging markets
# - 1d EMA(50) trend filter ensures alignment with higher timeframe direction
# - Position size: 0.25 discrete to minimize fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Works in bull/bear: Camarilla adapts to volatility, volume filter avoids false breakouts, trend filter avoids counter-trend trades

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar (based on previous bar)
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_12h = high_12h - low_12h
    camarilla_r4 = close_12h + 1.5 * range_12h
    camarilla_r3 = close_12h + 1.1 * range_12h
    camarilla_s3 = close_12h - 1.1 * range_12h
    camarilla_s4 = close_12h - 1.5 * range_12h
    
    # Align Camarilla levels to 6h timeframe (using previous completed 12h bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute 12h volume confirmation (volume > 1.5x 20-period average)
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume_12h > (1.5 * vol_ma_20_12h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm_12h.astype(float))
    
    # Pre-compute 1d EMA for trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Primary price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Get current levels
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Volume confirmation (aligned)
        vol_confirm = volume_confirm_aligned[i] > 0.5  # Boolean as float
        
        # Trend filter
        trend_long = curr_close > ema_50_1d_aligned[i]
        trend_short = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R4 with volume confirmation and uptrend
            if curr_close > r4 and vol_confirm and trend_long:
                position = 1
                signals[i] = 0.25
            # Short breakout: price < S4 with volume confirmation and downtrend
            elif curr_close < s4 and vol_confirm and trend_short:
                position = -1
                signals[i] = -0.25
            # Long fade: price < S3 with uptrend (mean reversion in uptrend)
            elif curr_close < s3 and trend_long:
                position = 1
                signals[i] = 0.25
            # Short fade: price > R3 with downtrend (mean reversion in downtrend)
            elif curr_close > r3 and trend_short:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: reverse signal or opposite Camarilla level touch
            if position == 1:  # Long position
                # Exit if price touches S3 (fade level) or gets bearish signal
                if curr_low <= s3 or (curr_close < r3 and not trend_long):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, short position
                # Exit if price touches R3 (fade level) or gets bullish signal
                if curr_high >= r3 or (curr_close > s3 and not trend_short):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals