#!/usr/bin/env python3
"""
Experiment #313: 1d Primary + 1w HTF — Fisher + Choppiness Regime Strategy

Hypothesis: Daily timeframe with weekly macro bias should produce fewer but higher-quality
trades than 4h strategies. Current best (#301) has Sharpe=0.612 on 4h. Moving to 1d should:
1. Reduce noise and whipsaws from lower timeframes
2. Capture major regime shifts more cleanly
3. Work better in bear/range markets (2025 test period) with Fisher Transform reversals

Key innovations vs #301:
- Ehlers Fisher Transform (period=9) for reversal detection — catches bear market rallies
- ADX(14) for trend strength confirmation — not just Choppiness
- Asymmetric logic: aggressive shorts in bear regime (price<1w HMA), conservative longs
- Wider stops (3x ATR vs 2.5x) appropriate for daily timeframe
- Position size: 0.25 (conservative for 1d, targets 20-40 trades/year)

Regime logic:
- CHOP > 55 + ADX < 20 = RANGE → Connors RSI mean reversion
- CHOP < 45 + ADX > 25 = TREND → Fisher Transform breakouts + HMA bias
- Neutral zone uses trend logic as default

TARGET: Sharpe > 0.612 (beat #301), 20-40 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_choppiness_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0)
    
    # RSI Streak (2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    gain_streak = streak_s.diff().clip(lower=0)
    loss_streak = (-streak_s.diff()).clip(lower=0)
    avg_gain_streak = gain_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss_streak = loss_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + rs_streak))
    streak_rsi = streak_rsi.fillna(50.0)
    streak_rsi = np.where(delta > 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    return crsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (2 * (close - LL) / (HH - LL) - 1)
    Signal line = Fisher shifted by 1
    """
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Normalize price position within range
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 0.67 * (2 * (close - lowest) / (highest - lowest + 1e-10) - 1)
        x = np.clip(x, -0.999, 0.999)  # Prevent division by zero in ln
        fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    
    fisher = np.nan_to_num(fisher, nan=0.0)
    return fisher

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1d timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Fisher Transform signal line (shifted by 1)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # RANGE: CHOP > 55 AND ADX < 20
        # TREND: CHOP < 45 AND ADX > 25
        is_range = chop[i] > 55.0 and adx_14[i] < 20.0
        is_trend = chop[i] < 45.0 and adx_14[i] > 25.0
        # Neutral: use trend logic as default
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_range:
            # RANGE REGIME: Connors RSI Mean Reversion
            # Long: CRSI < 15 (oversold) + price above weekly HMA (bullish bias)
            # Short: CRSI > 85 (overbought) + price below weekly HMA (bearish bias)
            if crsi[i] < 15.0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
            elif crsi[i] > 85.0 and price_below_hma_1w:
                desired_signal = -POSITION_SIZE
        
        else:  # is_trend or neutral
            # TREND REGIME: Fisher Transform Reversals + Macro Bias
            # ASYMMETRIC: More aggressive shorts in bear market (price < 1w HMA)
            
            # LONG: Fisher crosses above -1.5 (oversold reversal) + bullish macro
            # Use fisher_signal[i] (previous bar) for crossover detection
            if fisher[i] > -1.5 and fisher_signal[i] <= -1.5 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
            # SHORT: Fisher crosses below +1.5 (overbought reversal) + bearish macro
            elif fisher[i] < 1.5 and fisher_signal[i] >= 1.5 and price_below_hma_1w:
                # More aggressive short sizing in bear regime
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (3.0 * ATR trailing for daily) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MACRO BIAS REVERSAL EXIT ===
        # Exit long if price crosses below 1w HMA (major trend change)
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        # Exit short if price crosses above 1w HMA
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit in range regime) ===
        if is_range and in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if is_range and in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (take profit in trend regime) ===
        if not is_range and in_position and position_side > 0 and fisher[i] > 2.0:
            desired_signal = 0.0
        
        if not is_range and in_position and position_side < 0 and fisher[i] < -2.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Maintain position if macro bias still supports it
            if position_side > 0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals