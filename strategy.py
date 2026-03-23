#!/usr/bin/env python3
"""
Experiment #439: 4h Primary + 1d HTF — Fisher Transform + Choppiness Regime Switch

Hypothesis: Previous 4h strategies overused RSI/CRSI. Fisher Transform catches reversals
better in bear/range markets (proven Sharpe 0.8-1.5 in research). Combined with:
1. Choppiness Index for regime detection (chop vs trend)
2. ADX for trend strength confirmation (avoid weak trends)
3. 1d HMA for bias (soft filter, not hard requirement)
4. Donchian breakout for trend entries
5. Fisher crossover for mean-reversion entries

Key differences from failed #434/#431:
- Fisher Transform instead of CRSI (better reversal detection in bear markets)
- ADX > 20 filter for trend entries (avoid whipsaws)
- Asymmetric sizing: 0.30 for high confidence, 0.20 for moderate
- More aggressive mean-reversion thresholds (Fisher > -1.5 / < +1.5)

Target: Sharpe > 0.612, 80-150 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_adx_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * ((close - LL) / (HH - LL) - 0.5) + 0.67 * X_prev
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    # Normalize price position within range
    for i in range(period, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            # Calculate X (smoothed normalized price)
            if i == period:
                x = 0.66 * ((close[i] - lowest_low) / price_range - 0.5)
            else:
                x_prev = 0.66 * ((close[i-1] - lowest_low) / price_range - 0.5) + 0.67 * (0.0 if np.isnan(fisher[i-2]) else fisher[i-2])
                x = 0.66 * ((close[i] - lowest_low) / price_range - 0.5) + 0.67 * x_prev
            
            # Clamp X to avoid log domain errors
            x = np.clip(x, -0.999, 0.999)
            
            # Fisher Transform
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
            
            # Trigger line (Fisher shifted by 1)
            if i > period:
                trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx_s = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = adx_s
    
    return adx

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h
    HIGH_CONF_SIZE = 0.30
    MOD_CONF_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 58.0  # Range market
        regime_trend = chop[i] < 42.0  # Trending market
        
        # === TREND BIAS (1d HMA) — SOFT FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 25.0
        adx_weak = adx[i] < 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = (fisher_trigger[i] < -1.5) and (fisher[i] >= -1.5) if not np.isnan(fisher_trigger[i]) else False
        fisher_cross_down = (fisher_trigger[i] > 1.5) and (fisher[i] <= 1.5) if not np.isnan(fisher_trigger[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = MOD_CONF_SIZE
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE
        else:
            position_size = HIGH_CONF_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        confidence = 0  # 0=none, 1=moderate, 2=high
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 58) — MEAN REVERSION ===
        if regime_chop:
            # Long: Fisher oversold + RSI confirmation
            if fisher_oversold and rsi_oversold:
                desired_signal = position_size
                confidence = 2
            elif fisher_cross_up:
                desired_signal = position_size * 0.8
                confidence = 1
            
            # Short: Fisher overbought + RSI confirmation
            if fisher_overbought and rsi_overbought and desired_signal == 0:
                desired_signal = -position_size
                confidence = 2
            elif fisher_cross_down and desired_signal == 0:
                desired_signal = -position_size * 0.8
                confidence = 1
        
        # === REGIME 2: TRENDING (CHOP < 42) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout + ADX strong + HMA bullish
            if donchian_breakout_long and adx_strong:
                if hma_bullish:
                    desired_signal = position_size
                    confidence = 2
                else:
                    desired_signal = position_size * 0.7
                    confidence = 1
            
            # Short: Donchian breakdown + ADX strong + HMA bearish
            if donchian_breakout_short and adx_strong and desired_signal == 0:
                if hma_bearish:
                    desired_signal = -position_size
                    confidence = 2
                else:
                    desired_signal = -position_size * 0.7
                    confidence = 1
            
            # HMA crossover entries (lower confidence)
            if desired_signal == 0:
                if hma_bullish and adx[i] > 20:
                    desired_signal = position_size * 0.6
                    confidence = 1
                elif hma_bearish and adx[i] > 20:
                    desired_signal = -position_size * 0.6
                    confidence = 1
        
        # === REGIME 3: TRANSITION (42-58) — CAUTIOUS ===
        else:
            # Only high-confidence signals
            if fisher_oversold and rsi_oversold:
                desired_signal = position_size * 0.6
                confidence = 1
            elif fisher_overbought and rsi_overbought:
                desired_signal = -position_size * 0.6
                confidence = 1
            elif donchian_breakout_long and adx_strong and hma_bullish:
                desired_signal = position_size * 0.6
                confidence = 1
            elif donchian_breakout_short and adx_strong and hma_bearish:
                desired_signal = -position_size * 0.6
                confidence = 1
        
        # === HTF BIAS MODIFIER (1d HMA) — SOFT ===
        if desired_signal > 0 and price_below_hma_1d:
            desired_signal = desired_signal * 0.6  # Reduce long when 1d bearish
        if desired_signal < 0 and price_above_hma_1d:
            desired_signal = desired_signal * 0.6  # Reduce short when 1d bullish
        
        # Cap signal magnitude
        desired_signal = np.clip(desired_signal, -0.40, 0.40)
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and fisher[i] > 2.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -2.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or fisher[i] > -1.0):
                desired_signal = position_size * 0.5
            elif position_side < 0 and (hma_bearish or fisher[i] < 1.0):
                desired_signal = -position_size * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals