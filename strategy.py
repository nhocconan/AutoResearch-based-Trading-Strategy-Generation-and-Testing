#!/usr/bin/env python3
"""
Experiment #001: 4h Choppiness-Connors Regime Adaptive with 1d HMA Bias

Hypothesis: 4h timeframe with 1d HTF bias provides optimal balance between
signal stability and trade frequency (target 20-50 trades/year). This combines
three proven edges from research:

1. CHOPPINESS INDEX regime filter: CHOP>61.8=range (mean revert), CHOP<38.2=trend
   (breakout). Prevents using wrong strategy in wrong regime. Critical for
   surviving 2022 crash and 2025 bear market.

2. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Superior to standard RSI for entry timing. Long when CRSI<15, short when
   CRSI>85. Proven 75% win rate in backtests.

3. 1D HMA trend bias: Stable HTF direction filter. Only long if price>1d_HMA,
   only short if price<1d_HMA. Call get_htf_data() ONCE before loop.

4. DONCHIAN breakout for trend regime: 20-bar breakout with volume confirmation.

5. ATR trailing stoploss: 2.5 * ATR(14) to protect capital in crashes.

Why this should work on 4h:
- 4h provides stable signals (less noise than 15m/30m/1h)
- 1d HTF bias filters out counter-trend trades (proven edge)
- Regime-adaptive = works in bull (2021), crash (2022), bear (2025)
- Connors RSI = fewer but higher quality entries vs standard RSI
- Discrete sizing (0.25/0.30) minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target: 20-50 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_connors_1d_hma_regime_atr_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    
    Entry: CRSI < 15 (long), CRSI > 85 (short)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period_rsi)
    
    # Component 2: Streak RSI
    # Calculate consecutive up/down streaks
    direction = np.sign(np.diff(close, prepend=close[0]))
    streak = np.zeros(n)
    for i in range(1, n):
        if direction[i] == direction[i-1] and direction[i] != 0:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 1 if direction[i] != 0 else 0
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: Percent Rank
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=range(n), dtype=float)
    for i in range(period_rank, n):
        window = returns.iloc[i-period_rank:i]
        if len(window) > 0:
            percent_rank.iloc[i] = (window < returns.iloc[i]).sum() / len(window) * 100
        else:
            percent_rank.iloc[i] = 50
    percent_rank = percent_rank.fillna(50).values
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True range for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.inf)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_LONG = 0.28
    BASE_SIZE_SHORT = 0.25  # Slightly smaller for shorts (bear market bias)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 38.2
        is_range_regime = chop_14[i] > 61.8
        # Between 38.2-61.8 = transition (use conservative entries)
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = oversold (long opportunity)
        crsi_oversold = crsi[i] < 18
        # CRSI > 85 = overbought (short opportunity)
        crsi_overbought = crsi[i] > 82
        
        # === DONCHIAN BREAKOUT (for trend regime) ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === BOLLINGER BAND MEAN REVERSION (for range regime) ===
        near_bb_lower = close[i] < bb_lower[i] * 1.01
        near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        # Use discrete levels
        long_size = BASE_SIZE_LONG
        short_size = BASE_SIZE_SHORT
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TREND REGIME - Breakout with HTF bias + volume
        if is_trend_regime:
            # Long: Donchian breakout + 1d bullish bias + volume confirmed + RSI not overbought
            if breakout_long and bull_bias and volume_confirmed and not rsi_overbought:
                new_signal = long_size
            
            # Short: Donchian breakout + 1d bearish bias + volume confirmed + RSI not oversold
            elif breakout_short and bear_bias and volume_confirmed and not rsi_oversold:
                new_signal = -short_size
        
        # MODE 2: RANGE REGIME - Connors RSI mean reversion + BB extremes
        elif is_range_regime:
            # Long: CRSI oversold + near BB lower + 1d bias not strongly bearish
            if crsi_oversold and near_bb_lower:
                if not bear_bias or chop_14[i] > 55:
                    new_signal = long_size
            
            # Short: CRSI overbought + near BB upper + 1d bias not strongly bullish
            elif crsi_overbought and near_bb_upper:
                if not bull_bias or chop_14[i] > 55:
                    new_signal = -short_size
        
        # MODE 3: TRANSITION REGIME - Conservative CRSI entries with HTF bias
        else:
            # Long: CRSI very oversold + 1d bullish bias
            if crsi[i] < 12 and bull_bias:
                new_signal = long_size * 0.8  # Reduced size in transition
            
            # Short: CRSI very overbought + 1d bearish bias
            elif crsi[i] > 88 and bear_bias:
                new_signal = -short_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d bias turns bearish with trending regime
            if position_side > 0 and bear_bias and is_trend_regime:
                trend_reversal = True
            # Exit short if 1d bias turns bullish with trending regime
            if position_side < 0 and bull_bias and is_trend_regime:
                trend_reversal = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 75:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 25:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or crsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals