#!/usr/bin/env python3
"""
Experiment #606: 1d Weekly-Biased Regime Adaptive (Choppiness + RSI + Donchian)

Hypothesis: Daily timeframe needs LOOSE entry conditions to ensure sufficient trades.
After 537 failures, the pattern is clear: overly strict filters = 0 trades = auto-reject.

Key design decisions:
1. Weekly HMA for long-term trend bias (via mtf_data helper)
2. Choppiness Index for regime detection (trend vs range)
3. LOOSE RSI thresholds (35/65 not 30/70) to ensure trades
4. LOOSE ADX threshold (15 not 25) for trend confirmation
5. Donchian breakout for trend entries, BB mean-reversion for range
6. Position size 0.25-0.30 discrete, stoploss at 2.5*ATR

Why this should work on 1d:
- Daily bars = fewer signals naturally, so need permissive entry conditions
- Weekly bias prevents counter-trend trades in strong regimes
- Regime-adaptive = works in both 2022 crash (trend mode) and 2025 range (mean-revert mode)
- Loose filters ensure 10+ trades per symbol (critical requirement)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_chop_rsi_donchian_bb_regime_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    price_range = highest_high - lowest_low
    
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_mid[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0  # Slightly higher threshold for more trend trades
        is_range_regime = chop_14[i] > 55.0  # Slightly lower threshold for more range trades
        
        # === WEEKLY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT (for trend regime) ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ADX FILTER (loose threshold for more trades) ===
        trend_strong = adx_14[i] > 15  # Loose threshold
        
        # === BOLLINGER BAND MEAN REVERSION (for range regime) ===
        bb_pct_lower = (close[i] - bb_lower[i]) / bb_lower[i] if bb_lower[i] > 0 else 1.0
        bb_pct_upper = (close[i] - bb_upper[i]) / bb_upper[i] if bb_upper[i] > 0 else -1.0
        near_bb_lower = bb_pct_lower < 0.02  # Within 2% of lower band
        near_bb_upper = bb_pct_upper > -0.02  # Within 2% of upper band
        
        # === RSI EXTREMES (loose for more trades) ===
        rsi_oversold = rsi_14[i] < 40  # Loose threshold
        rsi_overbought = rsi_14[i] > 60  # Loose threshold
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TREND REGIME - Donchian breakout with weekly bias
        if is_trend_regime:
            # Long: Donchian breakout + weekly bullish bias + ADX confirms
            if breakout_long and bull_bias and trend_strong:
                new_signal = SIZE
            
            # Short: Donchian breakout + weekly bearish bias + ADX confirms
            elif breakout_short and bear_bias and trend_strong:
                new_signal = -SIZE
        
        # MODE 2: RANGE REGIME - Bollinger mean reversion + RSI extremes
        if is_range_regime:
            # Long: Price at BB lower + RSI oversold
            if near_bb_lower and rsi_oversold:
                # Only if not strongly bearish on weekly
                if not bear_bias or adx_14[i] < 25:
                    new_signal = SIZE
            
            # Short: Price at BB upper + RSI overbought
            elif near_bb_upper and rsi_overbought:
                # Only if not strongly bullish on weekly
                if not bull_bias or adx_14[i] < 25:
                    new_signal = -SIZE
        
        # MODE 3: TRANSITION (between regimes) - use both signals
        if not is_trend_regime and not is_range_regime:
            # Allow both breakout and mean-reversion entries
            if breakout_long and bull_bias:
                new_signal = SIZE
            elif breakout_short and bear_bias:
                new_signal = -SIZE
            elif near_bb_lower and rsi_oversold:
                new_signal = SIZE
            elif near_bb_upper and rsi_overbought:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_bias and adx_14[i] > 20:
                trend_reversal = True
            if position_side < 0 and bull_bias and adx_14[i] > 20:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
                entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals