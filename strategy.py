#!/usr/bin/env python3
"""
Experiment #3620: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. HMA(21) filters for trend direction to avoid counter-trend trades. Volume confirmation ensures breakout validity. Works in bull markets (breakouts to new highs) and bear markets (breakdowns to new lows). Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year). Uses 1d for HTF trend context and 4h for entry timing and risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3620_4h_donchian20_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend context (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators: Donchian Channel(20) ===
    lookback_donchian = 20
    highest_high = pd.Series(high).rolling(window=lookback_donchian, min_periods=lookback_donchian).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_donchian, min_periods=lookback_donchian).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # === 4h Indicators: HMA(21) for trend ===
    def calculate_hma(series, period):
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
        hma = pd.Series(2 * wma_half - wma_full).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    hma_21 = calculate_hma(close, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)  # Using 1d dataframe for alignment (same length as prices after shift)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for volatility and stoploss ===
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_donchian + 1, 50, 21, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(hma_21_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower (20-period) - trend reversal
                elif price < donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper (20-period) - trend reversal
                elif price > donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) for breakout validity
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Determine trend bias from 1d EMA(50) and HMA(21)
            bullish_bias = ema_1d_aligned[i] > close_1d[-1] if len(close_1d) > 0 else price > hma_21_aligned[i]
            
            # Long entry: Price breaks above Donchian upper in bullish trend
            if (price > donchian_upper[i] and 
                bullish_bias and
                hma_21_aligned[i] > hma_21_aligned[i-1]):  # HMA rising
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower in bearish trend
            elif (price < donchian_lower[i] and 
                  not bullish_bias and
                  hma_21_aligned[i] < hma_21_aligned[i-1]):  # HMA falling
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