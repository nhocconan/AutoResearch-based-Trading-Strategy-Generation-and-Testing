#!/usr/bin/env python3
"""
Experiment #007: 6h Williams Alligator + Elder Ray + ADX Regime

HYPOTHESIS: 
- Williams Alligator provides trend structure (Jaw/Teeth/Lips alignment)
- Elder Ray measures buying/selling pressure relative to EMA(13)
- ADX regime filter prevents trading in choppy/ranging markets (the #1 killer)
- 6h balances between too few trades (12h/1d) and overtrading (4h)

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Alligator张嘴 (Lips > Teeth > Jaw) + Bull Power > 0 + ADX > 25
- Bear: Alligator闭嘴 (Lips < Teeth < Jaw) + Bear Power < 0 + ADX > 25
- Range: ADX < 20 = no trades (avoid whipsaws)

EXPECTED TRADES: 75-150 total over 4 years (19-37/year)
- Donchian(20) breakout every ~30 bars = 350/year potential
- ADX filter (50% reduction) = 175/year
- Alligator alignment (40%) = 105/year
- Volume spike (30%) = 75/year final
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_alligator_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """ADX + DI+/DI- for regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+/DI-
    di_plus = np.where(atr_smooth > 0, 100 * plus_dm_smooth / atr_smooth, 0)
    di_minus = np.where(atr_smooth > 0, 100 * minus_dm_smooth / atr_smooth, 0)
    
    # DX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    
    return adx, di_plus, di_minus

def calculate_alligator(high, low, close):
    """
    Williams Alligator: SMMA on median price, shifted
    Jaw = SMA(13) shifted 8
    Teeth = SMA(8) shifted 5
    Lips = SMA(5) shifted 3
    """
    median = (high + low + close) / 2.0
    
    # SMMA (smoothed moving average) = EMA for our purposes
    jaw = pd.Series(median).ewm(span=13, min_periods=13, adjust=False).mean().values
    teeth = pd.Series(median).ewm(span=8, min_periods=8, adjust=False).mean().values
    lips = pd.Series(median).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # Shift forward (price action to the right of indicator lines)
    jaw_shifted = pd.Series(jaw).shift(8).values
    teeth_shifted = pd.Series(teeth).shift(5).values
    lips_shifted = pd.Series(lips).shift(3).values
    
    return jaw_shifted, teeth_shifted, lips_shifted

def calculate_elder_ray(high, low, close, ema_period=13):
    """Elder Ray: measures buying/selling pressure"""
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power, ema

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX for regime (trend vs choppy)
    adx_1d, di_plus_1d, di_minus_1d = calculate_adx(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Elder Ray
    bull_power, bear_power, ema_13 = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Donchian for breakout
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for 6h
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 80  # Enough for Alligator shifts (8+5+3=8) + Donchian(20) + ADX(14)
    
    for i in range(warmup, n):
        # === NaN checks ===
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: ADX from 1d ===
        adx_regime = adx_1d_aligned[i]
        is_trending = adx_regime > 20  # Below 20 = choppy, no trades
        
        # === ALLIGATOR TREND ===
        jaw_val = jaw[i] if not np.isnan(jaw[i]) else 0
        teeth_val = teeth[i] if not np.isnan(teeth[i]) else 0
        lips_val = lips[i] if not np.isnan(lips[i]) else 0
        
        # Bullish: Lips > Teeth > Jaw (Alligator eating = uptrend)
        alligator_bull = lips_val > teeth_val > jaw_val
        # Bearish: Lips < Teeth < Jaw (Alligator eating = downtrend)
        alligator_bear = lips_val < teeth_val < jaw_val
        
        # === ELDER RAY CONFIRMATION ===
        bull_ok = bull_power[i] > 0  # Buyers pushing above EMA
        bear_ok = bear_power[i] < 0  # Sellers pushing below EMA
        
        # === DONCHIAN BREAKOUT ===
        prev_upper = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_upper) and high[i] > prev_upper)
        bearish_breakout = (not np.isnan(prev_lower) and low[i] < prev_lower)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and is_trending:
            # LONG: Breakout + Alligator bullish + Elder Ray confirms + Volume
            if bullish_breakout and alligator_bull and bull_ok and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Breakdown + Alligator bearish + Elder Ray confirms + Volume
            elif bearish_breakout and alligator_bear and bear_ok and vol_spike:
                desired_signal = -SIZE
        
        # === EXIT / STOP LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from entry (wider for 6h)
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend weakens (ADX drops or Alligator flattens)
                elif adx_regime < 18 or (not alligator_bull and not alligator_bear):
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from entry
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend weakens
                elif adx_regime < 18 or (not alligator_bull and not alligator_bear):
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === EXECUTE ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals