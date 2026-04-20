#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 12h trend filter and volume confirmation.
# Fades at R3/S3 levels when 12h EMA50 aligns with reversal direction.
# Uses volume spike (>2x 20-bar avg) to confirm institutional participation.
# Works in bull/bear by following higher timeframe trend while exploiting intraday mean reversion at extreme levels.
# Target: 20-40 trades per year to minimize fee drag.

name = "6h_Camarilla_R3S3_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h EMA50 for trend direction ===
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === Camarilla levels from previous day (using daily OHLC) ===
    # We'll approximate using 12h data since we don't have direct daily in this context
    # Alternative: use rolling window on 6h data to get daily-like OHLC
    # But per rules, we should use actual daily data if needed - however we're using 12h as HTF
    # Let's use 12h high/low/close to approximate daily Camarilla
    
    # For Camarilla, we need previous period's OHLC
    # Since we're on 6h timeframe, we'll use 12h as our "daily" equivalent for pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate typical Camarilla levels based on previous 12h bar
    # H, L, C from previous bar
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan  # First value invalid
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, np.nan, range_val)
    
    # Resistance levels
    R3 = prev_close + range_val * 1.1 / 4
    R4 = prev_close + range_val * 1.1 / 2
    # Support levels
    S3 = prev_close - range_val * 1.1 / 4
    S4 = prev_close - range_val * 1.1 / 2
    
    # Align to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    
    # === 6h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_aligned[i]
        r3_val = R3_aligned[i]
        r4_val = R4_aligned[i]
        s3_val = S3_aligned[i]
        s4_val = S4_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(r3_val) or np.isnan(r4_val) or 
            np.isnan(s3_val) or np.isnan(s4_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price at S3/S4 with 12h uptrend and volume spike
            # Fade at S3 (strong support) or break above S4 with momentum
            if close_val <= s3_val and ema_val > close_val and vol_ratio_val > 2.0:
                # Fade long at S3 - expect bounce
                signals[i] = 0.25
                position = 1
            elif close_val >= s4_val and ema_val > close_val and vol_ratio_val > 2.0:
                # Break above S4 with momentum in uptrend
                signals[i] = 0.25
                position = 1
            # Short setup: price at R3/R4 with 12h downtrend and volume spike
            elif close_val >= r3_val and ema_val < close_val and vol_ratio_val > 2.0:
                # Fade short at R3 - expect rejection
                signals[i] = -0.25
                position = -1
            elif close_val <= r4_val and ema_val < close_val and vol_ratio_val > 2.0:
                # Break below R4 with momentum in downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reverse signal or mean reversion
            # Exit when price reaches midpoint or shows rejection
            mid_point = (s3_val + r3_val) / 2  # Approximate midpoint
            if close_val >= mid_point or (close_val <= s3_val and vol_ratio_val > 2.0):
                # Either reached middle or strong rejection at support
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse signal or mean reversion
            mid_point = (s3_val + r3_val) / 2  # Approximate midpoint
            if close_val <= mid_point or (close_val >= r3_val and vol_ratio_val > 2.0):
                # Either reached middle or strong rejection at resistance
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals