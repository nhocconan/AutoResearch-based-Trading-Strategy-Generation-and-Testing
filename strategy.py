#!/usr/bin/env python3
"""
Experiment #007: 6h Elder Ray Momentum + EMA Crossover + Choppiness Regime

HYPOTHESIS: Elder Ray measures institutional "power" - how far price penetrates
above/below EMA. Combined with EMA crossover for direction and Choppiness to
avoid whipsaws, this captures multi-day moves with smart money confirmation.

WHY ELDER RAY IS DIFFERENT:
- TRIX measures rate of change (derivative) - this is second derivative
- Donchian measures pure price range breakout
- CRSI measures mean reversion extremes
- Elder Ray measures BULL/BEAR POWER relative to EMA (grounded in real price action)

WHY 6h:
- 4h overtraded historically (23% keep rate vs 6h's 23% but different strategies)
- 12h had some keepers (Sharpe -0.14 to -0.22) - similar approach on 6h
- Natural trade frequency: 75-150 total over 4 years (18-37/year)
- Captures multi-day institutional moves without excessive noise

WHY IT WORKS IN BULL + BEAR:
- Bull: BullPower positive + EMA crossover + in_trend regime → long
- Bear: BearPower negative + EMA cross down + in_trend regime → short
- Range: Choppiness>61.8 → no trades (avoid whipsaws)

TARGET: 75-150 total trades over 4 years.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_ema_chop_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, period, min_periods=None):
    """Calculate EMA with proper min_periods"""
    if min_periods is None:
        min_periods = period
    return pd.Series(values).ewm(span=period, min_periods=min_periods, adjust=False).mean().values

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

def calculate_elder_ray(close, high, low, ema_period=13):
    """
    Elder Ray: Measures bullish/bearish power
    Bull Power = High - EMA(close, period)
    Bear Power = Low - EMA(close, period)
    
    In uptrend: Bull Power > 0 (price above EMA, highs pushing higher)
    In downtrend: Bear Power < 0 (price below EMA, lows pushing lower)
    """
    n = len(close)
    if n < ema_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    ema_close = calculate_ema(close, ema_period, min_periods=ema_period)
    
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    return bull_power, bear_power

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for multi-timeframe trend (faster than EMA200, more signals)
    ema_50_1d = calculate_ema(df_1d['close'].values, 50, min_periods=50)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Local 6h indicators ===
    # EMA crossover for direction
    ema_8 = calculate_ema(close, 8, min_periods=8)
    ema_21 = calculate_ema(close, 21, min_periods=21)
    
    # Elder Ray (Bull/Bear Power) - uses EMA13
    bull_power, bear_power = calculate_elder_ray(close, high, low, ema_period=13)
    
    # ATR for stops
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness for regime filter
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signala ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Need at least EMA(21) + Elder Ray(13) + Choppiness(14)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION: EMA crossover ===
        ema8_above_ema21 = ema_8[i] > ema_21[i]
        ema8_below_ema21 = ema_8[i] < ema_21[i]
        
        # === ELDER RAY SIGNALS ===
        # Bull Power > 0: price pushing above EMA (bulls in control)
        # Bear Power < 0: price pushing below EMA (bears in control)
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # === HTF TREND: Price vs 1d EMA50 ===
        above_htf_ema = close[i] > ema_50_aligned[i]
        below_htf_ema = close[i] < ema_50_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        chop = chop_14[i]
        in_chop = chop > 61.8 if not np.isnan(chop) else False
        # Allow trades when trending (chop < 50) or neutral (50 <= chop <= 61.8)
        allow_trades = chop < 61.8 if not np.isnan(chop) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and allow_trades:
            # === LONG ENTRY ===
            # Bull market: EMA8 > EMA21 + Bull Power > 0 + above HTF EMA + volume
            # Key insight: Elder Ray confirms momentum behind the EMA crossover
            if ema8_above_ema21 and bull_strong and above_htf_ema and vol_spike:
                desired_signal = SIZE
            
            # Alternative: Strong bull power pullback to EMA21
            elif ema8_above_ema21 and above_htf_ema:
                if not np.isnan(bull_power[i-1]) and bull_power[i-1] < 0 and bull_strong:
                    # Bull power just turned positive - momentum shift
                    if vol_spike or vol_ratio[i] > 1.2:
                        desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Bear market: EMA8 < EMA21 + Bear Power < 0 + below HTF EMA + volume
            if ema8_below_ema21 and bear_strong and below_htf_ema and vol_spike:
                desired_signal = -SIZE
            
            # Alternative: Strong bear power bounce to EMA21
            elif ema8_below_ema21 and below_htf_ema:
                if not np.isnan(bear_power[i-1]) and bear_power[i-1] > 0 and bear_strong:
                    # Bear power just turned negative - momentum shift
                    if vol_spike or vol_ratio[i] > 1.2:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                # Long stop: entry - 2.5 ATR
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Take profit: exit if EMA crossover flips bearish
                elif ema8_below_ema21 and bars_held >= 2:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Trailing: allow 1 ATR buffer above entry
                elif close[i] < entry_price - 1.0 * entry_atr and ema8_below_ema21:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            elif position_side < 0:
                # Short stop: entry + 2.5 ATR
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Take profit: exit if EMA crossover flips bullish
                elif ema8_above_ema21 and bars_held >= 2:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Trailing: allow 1 ATR buffer below entry
                elif close[i] > entry_price + 1.0 * entry_atr and ema8_above_ema21:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
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