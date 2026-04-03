#!/usr/bin/env python3
"""
Experiment #068: 12h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian channel breakouts on 12h timeframe with HMA(21) trend filter and volume confirmation (>1.5x average) capture strong momentum moves in both bull and bear markets. Uses 1w HTF for regime filter (price > 1w HMA50 = bull bias, < = bear bias). Target: 50-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_068_12h_donchian_hma_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA(50) for bull/bear regime ===
    def calculate_hma(arr, period):
        """Hull Moving Average"""
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False, min_periods=half).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
        raw_hma = 2 * wma2 - wma1
        hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False, min_periods=sqrt).mean().values
        return hma
    
    hma_1w_50 = calculate_hma(df_1w['close'].values, 50)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # === 12h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: HMA(21) for trend filter ===
    hma_12h_21 = calculate_hma(close, 21)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 12h Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Warmup for Donchian, HMA, ATR stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_12h_21[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i]) or np.isnan(hma_1w_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        is_bull_regime = price > hma_1w_50_aligned[i]  # 1w HMA50 filter
        
        # --- Stoploss Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                if low[i] < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                if high[i] > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
        
        # --- New Position Entry Logic ---
        if not in_position:
            # Donchian breakout with volume confirmation and regime filter
            if price > donchian_upper[i-1] and vol_spike and hma_12h_21[i] > hma_12h_21[i-1]:
                # Bullish breakout: price above upper channel, rising HMA, volume spike
                if is_bull_regime or not is_bull_regime:  # Trade in both regimes
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    signals[i] = SIZE
            elif price < donchian_lower[i-1] and vol_spike and hma_12h_21[i] < hma_12h_21[i-1]:
                # Bearish breakout: price below lower channel, falling HMA, volume spike
                if is_bull_regime or not is_bull_regime:  # Trade in both regimes
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Hold position
            signals[i] = position_side * SIZE
    
    return signals