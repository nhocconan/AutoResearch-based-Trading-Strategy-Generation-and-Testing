#!/usr/bin/env python3
"""
Experiment #1983: 4h Donchian(20) Breakout + 12h/1d Trend Filter + Volume Confirmation
HYPOTHESIS: Donchian channel breakouts capture institutional order flow when aligned with 12h/1d trend and volume spikes. 
Strategy:
- Primary: 4h Donchian(20) breakout (long at upper band, short at lower band)
- HTF filters: 12h EMA(50) trend direction + 1d close > EMA(200) for regime filter
- Volume confirmation: 4h volume > 2.0x 20-period average to avoid false breakouts
- ATR stoploss: exit when price moves 2.5*ATR(14) against position
- Position size: 0.25 (discrete level to minimize fee churn)
Target: 75-200 total trades over 4 years (19-50/year) to avoid overtrading.
Works in bull/bear markets by requiring trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1983_4h_donchian20_12h_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA(50) trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_50_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === HTF: 1d data for regime filter (close > EMA(200)) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(200) for bull/bear regime
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    regime_1d = np.where(close_1d > ema_200_1d, 1, -1)  # 1=bull, -1=bear
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # === 4h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # sufficient for EMA(200) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(regime_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            stoploss_hit = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * atr[i]:
                    stoploss_hit = True
            else:  # Short position
                if price > entry_price + 2.5 * atr[i]:
                    stoploss_hit = True
            
            # Exit on Donchian opposite touch (mean reversion)
            donch_exit = False
            if position_side > 0 and price <= donch_low[i]:
                donch_exit = True
            elif position_side < 0 and price >= donch_high[i]:
                donch_exit = True
            
            if stoploss_hit or donch_exit:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h trend alignment and 1d regime filter
        trend_bias = trend_12h_aligned[i]
        regime_filter = regime_1d_aligned[i]
        
        # Volume confirmation: require significant spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper band AND 12h trend up AND 1d bull regime
            if trend_bias > 0 and regime_filter > 0 and price > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower band AND 12h trend down AND 1d bear regime
            elif trend_bias < 0 and regime_filter < 0 and price < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals