#!/usr/bin/env python3
"""
Experiment #070: 1d Donchian(20) Breakout + Weekly HMA Trend + Volume Spike

HYPOTHESIS: Donchian channel breakouts on daily timeframe, filtered by weekly HMA trend 
(price > weekly HMA21 = bullish bias, price < weekly HMA21 = bearish bias) and volume spikes 
(>1.8x 20-day average) capture strong momentum moves. Weekly HMA provides multi-timeframe 
trend alignment to avoid counter-trend breakouts. Daily timeframe targets 7-25 trades/year 
(30-100 total over 4 years) to minimize fee drag while capturing significant moves. 
Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. Works in both 
bull (breakouts with volume) and bear (failed breaks reverse sharply). Position sizing at 
0.25 to balance opportunity and risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_070_1d_donchian_weekly_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly HMA calculation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA(21) from 1w data
    def calculate_hma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(values).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(values).ewm(span=period, adjust=False).mean().values
        hma_values = 2 * wma_half - wma_full
        hma = pd.Series(hma_values).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    
    hma_21 = calculate_hma(df_1w['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # === 1d Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital) - balanced for drawdown control
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF weekly HMA, ATR, and volume
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly HMA Trend Filter: Price > HMA = bullish bias, Price < HMA = bearish bias ---
        price_above_weekly_hma = close[i] > hma_21_aligned[i]
        price_below_weekly_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]  # Wider stoploss for daily TF
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]  # Wider stoploss for daily TF
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above weekly HMA
        long_condition = breakout_up and volume_spike and price_above_weekly_hma
        
        # Short: Donchian breakout down + volume spike + price below weekly HMA
        short_condition = breakout_down and volume_spike and price_below_weekly_hma
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals