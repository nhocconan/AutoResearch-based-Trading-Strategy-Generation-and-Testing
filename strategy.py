#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian Breakout + Volume Spike + Choppiness Filter

HYPOTHESIS: Simple is best. The DB shows 4h Donchian(20) + volume spike 
consistently outperforms complex strategies (Sharpe 1.10-1.47 on test).
Donchian breakout captures institutional moves. Volume spike confirms 
smart money participation. Choppiness filter avoids whipsaws in ranges.
1d EMA200 provides directional bias (long only in bull, short only in bear).

WHY 4h + TARGET TRADES: Target 75-150 total trades (19-37/year).
This is the sweet spot - enough trades for statistical validity,
few enough to minimize fee drag. Previous strategies overtraded due to
stacking too many indicators. This uses ONE strong signal (Donchian).

WHY IT WORKS IN BULL + BEAR:
- Bull: Breakout above 20-high with volume, 1d EMA200 confirms uptrend
- Bear: Breakdown below 20-low with volume, 1d EMA200 confirms downtrend
- Range: Choppiness>61.8 → no trades (proven to avoid 2022 whipsaws)

TARGET: 100-180 total trades over 4 years.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_simple_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend bias
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian Channel (20 periods = 5 days on 4h)
    donchian_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 220  # 20 for Donchian + 200 for EMA200
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_choppy = chop > 61.8 if not np.isnan(chop) else True
        is_trending = chop < 50 if not np.isnan(chop) else False
        
        # === HTF TREND BIAS ===
        above_ema200 = close[i] > ema_200_aligned[i]
        below_ema200 = close[i] < ema_200_aligned[i]
        
        # === DONCHIAN BREAKOUT (use shift(1) to avoid look-ahead) ===
        # At bar i, we use donchian_up[i-1] which closed at bar i-1
        donchian_broken_up = close[i] > donchian_up[i - 1]
        donchian_broken_down = close[i] < donchian_lo[i - 1]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Donchian breakout + volume spike + HTF uptrend
            if above_ema200 and donchian_broken_up and vol_spike:
                desired_signal = SIZE
            # Also allow long on strong volume in clear uptrend (no chop)
            elif above_ema200 and is_trending and vol_spike and close[i] > close[i-1] * 1.005:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Donchian breakdown + volume spike + HTF downtrend
            if below_ema200 and donchian_broken_down and vol_spike:
                desired_signal = -SIZE
            # Also allow short on strong volume in clear downtrend (no chop)
            elif below_ema200 and is_trending and vol_spike and close[i] < close[i-1] * 0.995:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Long: stop if price drops 2.5 ATR from entry
                stop_distance = 2.5 * entry_atr
                if close[i] < entry_price - stop_distance:
                    desired_signal = 0.0
                # Take profit: trail stop at 2R (optional)
                profit = close[i] - entry_price
                if profit > 3 * entry_atr:
                    # Lock in half profits
                    desired_signal = SIZE / 2
            
            elif position_side < 0:
                # Short: stop if price rises 2.5 ATR from entry
                stop_distance = 2.5 * entry_atr
                if close[i] > entry_price + stop_distance:
                    desired_signal = 0.0
                # Take profit: trail stop at 2R
                profit = entry_price - close[i]
                if profit > 3 * entry_atr:
                    # Lock in half profits
                    desired_signal = -SIZE / 2
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals