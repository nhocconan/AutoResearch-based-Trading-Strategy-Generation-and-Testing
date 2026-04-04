#!/usr/bin/env python3
"""
Experiment #2318: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts with weekly trend alignment and volume confirmation
capture strong momentum moves while avoiding false breakouts. Works in bull markets
via breakouts and bear markets via short breakdowns. Weekly trend filter ensures
we only trade with the dominant higher timeframe momentum.
- Primary: 1d Donchian(20) breakout (long at 20-day high, short at 20-day low)
- HTF: 1w HMA(21) for trend alignment (only trade long when weekly HMA rising,
  short when falling)
- Entry: Long when price breaks above 20-day high + volume spike + weekly HMA up;
  Short when price breaks below 20-day low + volume spike + weekly HMA down
- Exit: ATR(14) stoploss (2*ATR) or opposite Donchian band touch
- Volume: Require > 1.5x 20-bar average spike to confirm participation
- Target: 30-100 total trades over 4 years (7-25/year) - suitable for 1d timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2318_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w HMA(21)
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        hma = 2 * wma_half - wma_full
        hma = pd.Series(hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    
    hma_1w = calculate_hma(close_1w, 21)
    # Trend: 1 when HMA rising, -1 when falling
    hma_diff = np.diff(hma_1w, prepend=hma_1w[0])
    trend_1w = np.where(hma_diff > 0, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian channels: 20-period high/low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 20  # sufficient for Donchian and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price moves 2*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches or crosses below Donchian low (contrarian exit)
                elif price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches or crosses above Donchian high (contrarian exit)
                elif price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        trend_bias = trend_1w_aligned[i]  # 1 for uptrend, -1 for downtrend
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above 20-day high + weekly uptrend
            if trend_bias > 0 and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below 20-day low + weekly downtrend
            elif trend_bias < 0 and price < donchian_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals