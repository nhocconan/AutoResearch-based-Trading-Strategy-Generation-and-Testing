#!/usr/bin/env python3
"""
Experiment #3851: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for continuation) 
capture institutional order flow. Volume > 1.8x MA(20) confirms participation. 
Works in bull/bear: In uptrend, buy R4 breakouts; in downtrend, short S4 breakdowns. 
In ranging markets, fade R3/S3 reversals. Discrete sizing (0.25) limits fee drag. 
ATR(14) trailing stop (2.0x) manages risk. Target: 75-150 trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3851_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Camarilla: Range = (H-L), then levels based on Close
    H_minus_L = high_1d - low_1d
    camarilla_r3 = close_1d + H_minus_L * 1.1 / 4
    camarilla_s3 = close_1d - H_minus_L * 1.1 / 4
    camarilla_r4 = close_1d + H_minus_L * 1.1 / 2
    camarilla_s4 = close_1d - H_minus_L * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine market regime based on price vs Camarilla levels
            # If price between R3 and S3: ranging market -> fade extremes
            # If price > R4: bullish breakout -> buy breakouts
            # If price < S4: bearish breakdown -> short breakdowns
            
            camarilla_r3 = camarilla_r3_aligned[i]
            camarilla_s3 = camarilla_s3_aligned[i]
            camarilla_r4 = camarilla_r4_aligned[i]
            camarilla_s4 = camarilla_s4_aligned[i]
            
            # Long entry conditions
            long_signal = False
            if price > camarilla_r4 and price > highest_high[i-1]:
                # Bullish breakout above R4
                long_signal = True
            elif camarilla_s3 < price < camarilla_r3 and price < camarilla_r3 and price > lowest_low[i-1]:
                # Fade from S3 in ranging market (reversal long)
                long_signal = True
            
            # Short entry conditions
            short_signal = False
            if price < camarilla_s4 and price < lowest_low[i-1]:
                # Bearish breakdown below S4
                short_signal = True
            elif camarilla_s3 < price < camarilla_r3 and price > camarilla_s3 and price < highest_high[i-1]:
                # Fade from R3 in ranging market (reversal short)
                short_signal = True
            
            if long_signal and not short_signal:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_signal and not long_signal:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals