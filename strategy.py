#!/usr/bin/env python3
"""
Experiment #3215: 6h Camarilla Pivot Reversal with 1d EMA Trend Filter and Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from daily timeframe act as strong support/resistance on 6h charts. 
Price rejecting at R3/S3 with 1d EMA alignment and volume spike indicates high-probability reversal. 
Breakout through R4/S4 with volume and EMA alignment suggests continuation. 
Designed to work in both bull (breakout continuation) and bear (reversal from extremes) markets via pivot structure.
Uses 6h timeframe for entries, 1d for pivot/EMA/volume filters. Target: 75-150 total trades over 4 years (19-37/year).
Position size 0.25. ATR-based stoploss (2.0x) manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3215_6h_camarilla_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots, EMA, volume (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(volume_1d))
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # Camarilla: 
    #   R4 = close + 1.5*(high - low)
    #   R3 = close + 1.1*(high - low)
    #   S3 = close - 1.1*(high - low)
    #   S4 = close - 1.5*(high - low)
    # We use prior day's OHLC to avoid look-ahead
    prior_close = np.concatenate([[np.nan], close_1d[:-1]])
    prior_high = np.concatenate([[np.nan], high_1d[:-1]])
    prior_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    r4 = prior_close + 1.5 * (prior_high - prior_low)
    r3 = prior_close + 1.1 * (prior_high - prior_low)
    s3 = prior_close - 1.1 * (prior_high - prior_low)
    s4 = prior_close - 1.5 * (prior_high - prior_low)
    
    # Align all 1d levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for EMA(50), vol MA(20), ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Check stoploss hit
            if position_side > 0:  # Long
                if price <= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price >= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if not volume_spike:
            signals[i] = 0.0
            continue
            
        # 1d EMA trend filter
        price_vs_ema = price - ema_1d_aligned[i]
        
        # Long setup: price at or below S3 with bullish EMA alignment
        if price <= s3_aligned[i] and price_vs_ema > 0:
            # Reversal long: enter at S3 test
            in_position = True
            position_side = 1
            entry_price = price
            stoploss_price = price - 2.0 * atr[i]  # 2*ATR stop
            signals[i] = SIZE
        # Short setup: price at or above R3 with bearish EMA alignment
        elif price >= r3_aligned[i] and price_vs_ema < 0:
            # Reversal short: enter at R3 test
            in_position = True
            position_side = -1
            entry_price = price
            stoploss_price = price + 2.0 * atr[i]  # 2*ATR stop
            signals[i] = -SIZE
        # Long breakout: price breaks above R4 with bullish EMA and volume
        elif price > r4_aligned[i] and price_vs_ema > 0:
            in_position = True
            position_side = 1
            entry_price = price
            stoploss_price = price - 2.0 * atr[i]
            signals[i] = SIZE
        # Short breakout: price breaks below S4 with bearish EMA and volume
        elif price < s4_aligned[i] and price_vs_ema < 0:
            in_position = True
            position_side = -1
            entry_price = price
            stoploss_price = price + 2.0 * atr[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals