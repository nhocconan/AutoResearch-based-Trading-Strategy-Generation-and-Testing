#!/usr/bin/env python3
"""
Experiment #194: 1h Camarilla Pivot + Volume Spike + 4h/1d Regime Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 1h combined with 4h/1d trend filters and volume confirmation. Uses higher timeframes for signal direction (4h/1d) and 1h only for entry timing precision. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years = 15-37/year for 1h. Discrete position sizing (0.20) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_194_1h_camarilla_pivot_volume_4h_1d_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for primary trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # EMA50 on 4h for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up_4h = close_4h > ema50_4h
    trend_down_4h = close_4h < ema50_4h
    # Align to 1h timeframe
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    trend_down_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_down_4h)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA200 on 1d for long-term regime
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    regime_bull = close_1d > ema200_1d  # Bull regime: price above 200 EMA
    regime_bear = close_1d < ema200_1d  # Bear regime: price below 200 EMA
    # Align to 1h timeframe
    regime_bull_aligned = align_htf_to_ltf(prices, df_1d, regime_bull)
    regime_bear_aligned = align_htf_to_ltf(prices, df_1d, regime_bear)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr_1h = np.zeros(n)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 1h Indicators: Camarilla Pivot Levels from previous bar ===
    camarilla_r4 = np.zeros(n)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    camarilla_pivot = np.zeros(n)
    
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        camarilla_pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        range_ = prev_high - prev_low
        
        camarilla_r4[i] = camarilla_pivot[i] + range_ * 1.1 / 2.0
        camarilla_r3[i] = camarilla_pivot[i] + range_ * 1.1 / 4.0
        camarilla_s3[i] = camarilla_pivot[i] - range_ * 1.1 / 4.0
        camarilla_s4[i] = camarilla_pivot[i] - range_ * 1.1 / 2.0
    
    camarilla_r4[0] = camarilla_r3[0] = camarilla_pivot[0] = camarilla_s3[0] = camarilla_s4[0] = close[0]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size - discrete level to minimize fee churn
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # Sufficient warmup for EMA200
    
    for i in range(warmup, n):
        # Skip if any data is invalid
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_up_4h_aligned[i]) or np.isnan(trend_down_4h_aligned[i]) or
            np.isnan(regime_bull_aligned[i]) or np.isnan(regime_bear_aligned[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # === Exit Logic (ATR-based stoploss) ===
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]  # Wider stop for 1h volatility
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at R3 or continue if breaks R4 with volume in trending regime
                if price >= camarilla_r3[i]:
                    vol_spike = vol_ratio[i] > 2.0
                    if price >= camarilla_r4[i] and vol_spike and regime_bull_aligned[i]:
                        # Continue trend in bull regime with volume
                        signals[i] = SIZE
                    else:
                        # Take profit at R3
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at S3 or continue if breaks S4 with volume in trending regime
                if price <= camarilla_s3[i]:
                    vol_spike = vol_ratio[i] > 2.0
                    if price <= camarilla_s4[i] and vol_spike and regime_bear_aligned[i]:
                        # Continue trend in bear regime with volume
                        signals[i] = -SIZE
                    else:
                        # Take profit at S3
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            
            # Minimum holding period: 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # === New Position Entry Logic ===
        vol_spike = vol_ratio[i] > 2.0  # Require strong volume confirmation
        
        # Mean reversion at R3/S3 with volume spike
        # Long: Price rejects R3 (comes back below) in bull regime with volume
        if (price < camarilla_r3[i] and 
            close[i-1] >= camarilla_r3[i-1] and  # Was at or above R3 previous bar
            regime_bull_aligned[i] and 
            vol_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price rejects S3 (goes back above) in bear regime with volume
        elif (price > camarilla_s3[i] and 
              close[i-1] <= camarilla_s3[i-1] and  # Was at or below S3 previous bar
              regime_bear_aligned[i] and 
              vol_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        # Breakout continuation at R4/S4 with volume spike and 4h trend alignment
        # Long: Price breaks above R4 with volume in 4h uptrend
        elif (price > camarilla_r4[i] and 
              trend_up_4h_aligned[i] and 
              vol_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below S4 with volume in 4h downtrend
        elif (price < camarilla_s4[i] and 
              trend_down_4h_aligned[i] and 
              vol_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals