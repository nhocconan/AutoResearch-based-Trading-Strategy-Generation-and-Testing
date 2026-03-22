#!/usr/bin/env python3
"""
Experiment #019: 4h Regime-Adaptive Strategy with Choppiness + Connors RSI + 1d HMA

Hypothesis: After 18 failures, return to proven regime-adaptive approach that switches
between mean reversion (in choppy markets) and trend following (in trending markets).
This combines:

1. Choppiness Index (14) - regime detection: CHOP > 61.8 = range, CHOP < 38.2 = trend
2. Connors RSI - mean reversion signal: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. 1d HMA(21) - major trend bias filter (only long if price > daily HMA)
4. HMA(21/48) crossover - trend following entry when in trend regime
5. ATR(14) trailing stop - 2.5 ATR exit to protect capital

Why this should work when others failed:
- Regime-adaptive: doesn't force trend logic in choppy markets (major failure mode)
- Connors RSI has 75% win rate in academic studies for mean reversion
- 4h timeframe = natural 20-50 trades/year (not too many fees)
- 1d HTF confirmation filters false signals during 2022-style crashes
- Discrete sizing (0.28/0.25) = minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.28 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_connors_1d_hma_regime_atr_v2"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI) - proven mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) - short-term momentum
    2. RSI on streak duration - measures consecutive up/down days
    3. PercentRank - current price vs last 100 prices
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            # Count consecutive up streaks in lookback
            up_count = 0
            for j in range(i - streak_period + 1, i + 1):
                if streak[j] >= 0:
                    up_count += 1
            streak_rsi[i] = (up_count / streak_period) * 100
        else:
            down_count = 0
            for j in range(i - streak_period + 1, i + 1):
                if streak[j] <= 0:
                    down_count += 1
            streak_rsi[i] = 100 - (down_count / streak_period) * 100
    
    # Component 3: PercentRank
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        lookback = close[i-pr_period+1:i+1]
        rank = np.sum(lookback <= close[i])
        percent_rank[i] = (rank / pr_period) * 100
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    # Fill early values
    crsi[:max(rsi_period, streak_period, pr_period)] = 50
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market (mean reversion works)
    - CHOP < 38.2 = trending market (trend following works)
    - 38.2 - 61.8 = transition zone
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr_values = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr_values[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_LONG = 0.28
    BASE_SIZE_SHORT = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION ===
        in_range_regime = chop[i] > 55  # Slightly lower threshold for more trades
        in_trend_regime = chop[i] < 45
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_21[i] > hma_4h_48[i]
        hma_bearish = hma_4h_21[i] < hma_4h_48[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CONNORS RSI MEAN REVERSION ===
        crsi_oversold = crsi[i] < 15  # Long signal
        crsi_overbought = crsi[i] > 85  # Short signal
        
        # === RSI EXTREMES (additional filter) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        long_size = BASE_SIZE_LONG
        short_size = BASE_SIZE_SHORT
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # LONG ENTRY
        if in_range_regime:
            # Mean reversion mode: Connors RSI oversold + daily bias
            if crsi_oversold and daily_bullish:
                new_signal = long_size
            elif crsi_oversold and rsi_oversold:
                # Strong mean reversion signal even without daily confirmation
                new_signal = long_size * 0.7
        elif in_trend_regime:
            # Trend following mode: HMA bullish + breakout or crossover
            if hma_bullish and daily_bullish:
                if breakout_long:
                    new_signal = long_size
                elif hma_4h_21[i] > hma_4h_48[i] and hma_4h_21[i-1] <= hma_4h_48[i-1]:
                    # HMA crossover
                    new_signal = long_size
        else:
            # Transition zone: require stronger confluence
            confluence = 0
            if hma_bullish:
                confluence += 1
            if daily_bullish:
                confluence += 1
            if crsi_oversold:
                confluence += 1
            if breakout_long:
                confluence += 1
            
            if confluence >= 3:
                new_signal = long_size * 0.8
        
        # SHORT ENTRY
        if in_range_regime:
            # Mean reversion mode: Connors RSI overbought + daily bias
            if crsi_overbought and daily_bearish:
                new_signal = -short_size
            elif crsi_overbought and rsi_overbought:
                new_signal = -short_size * 0.7
        elif in_trend_regime:
            # Trend following mode: HMA bearish + breakout or crossover
            if hma_bearish and daily_bearish:
                if breakout_short:
                    new_signal = -short_size
                elif hma_4h_21[i] < hma_4h_48[i] and hma_4h_21[i-1] >= hma_4h_48[i-1]:
                    new_signal = -short_size
        else:
            # Transition zone: require stronger confluence
            confluence = 0
            if hma_bearish:
                confluence += 1
            if daily_bearish:
                confluence += 1
            if crsi_overbought:
                confluence += 1
            if breakout_short:
                confluence += 1
            
            if confluence >= 3:
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and chop[i] > 65 and not crsi_oversold:
                # Long in newly choppy market without oversold signal
                regime_exit = True
            if position_side < 0 and chop[i] > 65 and not crsi_overbought:
                # Short in newly choppy market without overbought signal
                regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
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