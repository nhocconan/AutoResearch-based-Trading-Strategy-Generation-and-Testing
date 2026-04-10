#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# - Uses 1d Camarilla pivot levels (R3, S3, R4, S4) as significant support/resistance
# - Breakout above R3 or below S3 with 1d volume > 1.5x 20-day average signals continuation
# - Breakout above R4 or below S4 signals strong momentum (half position)
# - Works in both bull and bear markets by trading breakouts of key daily levels
# - Discrete position sizing (0.25/0.35) minimizes fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to avoid fee drag

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = camarilla_pivot + range_1d * 1.1 / 4.0
    camarilla_s3 = camarilla_pivot - range_1d * 1.1 / 4.0
    camarilla_r4 = camarilla_pivot + range_1d * 1.1 / 2.0
    camarilla_s4 = camarilla_pivot - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1d volume and its 20-day moving average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for 20-day volume MA
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_price = prices['close'].iloc[i]
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout above R3
            if close_price > camarilla_r3_aligned[i] and volume_confirm:
                # Strong breakout above R4
                if close_price > camarilla_r4_aligned[i]:
                    signals[i] = 0.35  # Half position for strong breakout
                    position = 1
                else:
                    signals[i] = 0.25  # Regular position for R3 breakout
                    position = 1
            # Short breakout below S3
            elif close_price < camarilla_s3_aligned[i] and volume_confirm:
                # Strong breakout below S4
                if close_price < camarilla_s4_aligned[i]:
                    signals[i] = -0.35  # Half position for strong breakout
                    position = -1
                else:
                    signals[i] = -0.25  # Regular position for S3 breakout
                    position = -1
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot level or opposite Camarilla level
            if position == 1:  # Long position
                # Exit if price drops back below R3 or reaches S3 (contrarian exit)
                if close_price < camarilla_r3_aligned[i] or close_price > camarilla_s3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25 if close_price <= camarilla_r4_aligned[i] else 0.35
            else:  # position == -1, short position
                # Exit if price rises back above S3 or reaches R3 (contrarian exit)
                if close_price > camarilla_s3_aligned[i] or close_price < camarilla_r3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25 if close_price >= camarilla_s4_aligned[i] else -0.35
    
    return signals