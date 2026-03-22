#!/usr/bin/env python3
"""
Experiment #059: 4h Primary + 1d HTF — Dual Regime Strategy

Hypothesis: Single-regime strategies fail because crypto alternates between trending
and ranging markets. This strategy uses Choppiness Index to detect regime and applies
different logic for each:

REGIME DETECTION:
- CHOP(14) > 61.8 = ranging market → use Connors RSI mean reversion
- CHOP(14) < 38.2 = trending market → use HMA trend following
- Between = neutral → reduce position size

ENTRY LOGIC:
- Range regime: Connors RSI < 15 (long) or > 85 (short) + price near BB bounds
- Trend regime: HMA(8/21) crossover + RSI pullback + 1d trend confirmation
- 1d HMA slope provides major bias (only trade with HTF trend in trend regime)

POSITION SIZING:
- Base: 0.30 discrete
- Reduce to 0.15 in neutral regime or weak signals
- Stoploss: 2.0 * ATR(14) trailing

Why this should work:
- Adapts to market conditions instead of forcing one approach
- Connors RSI has 75% win rate in ranges (research-backed)
- HMA trend following works in strong trends
- 1d HTF prevents counter-trend trades in major moves
- 4h timeframe naturally limits to 30-60 trades/year

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 base, 0.15 reduced
Stoploss: 2.0 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_connors_hma_1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.where(np.isnan(chop), 50, chop)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * min(streak_abs[i], streak_period) / streak_period
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 - min(streak_abs[i], streak_period) / streak_period)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100 * count_lower / rank_period
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    crsi = np.where(np.isnan(crsi), 50, crsi)
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # HMA for trend following
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range, CHOP < 38.2 = trend, between = neutral
        is_range_regime = chop_14[i] > 61.8
        is_trend_regime = chop_14[i] < 38.2
        is_neutral_regime = not is_range_regime and not is_trend_regime
        
        # === 1D TREND BIAS (MAJOR) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0
        trend_1d_bearish = hma_1d_slope_aligned[i] < 0
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === POSITION SIZING BY REGIME ===
        if is_trend_regime:
            current_size = BASE_SIZE
        elif is_range_regime:
            current_size = BASE_SIZE
        else:
            current_size = REDUCED_SIZE  # neutral = smaller positions
        
        # === RANGE REGIME: CONNORS RSI MEAN REVERSION ===
        range_signal = 0.0
        if is_range_regime:
            # Long: CRSI < 15 + price near lower BB
            if crsi[i] < 15 and close[i] < bb_lower[i] * 1.01:
                if trend_1d_bullish or not trend_1d_bearish:  # Avoid counter-trend
                    range_signal = current_size
            
            # Short: CRSI > 85 + price near upper BB
            if crsi[i] > 85 and close[i] > bb_upper[i] * 0.99:
                if trend_1d_bearish or not trend_1d_bullish:  # Avoid counter-trend
                    range_signal = -current_size
        
        # === TREND REGIME: HMA CROSSOVER + RSI PULLBACK ===
        trend_signal = 0.0
        if is_trend_regime:
            # HMA alignment
            hma_bullish = hma_8[i] > hma_21[i]
            hma_bearish = hma_8[i] < hma_21[i]
            
            # HMA crossover
            hma_bull_cross = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
            hma_bear_cross = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
            
            # RSI pullback levels (not extreme, allows more entries)
            rsi_pullback_long = 40 < rsi_14[i] < 60
            rsi_pullback_short = 40 < rsi_14[i] < 60
            
            # Long: bullish trend + 1d confirmation + pullback or crossover
            if trend_1d_bullish and hma_bullish:
                if hma_bull_cross:
                    trend_signal = current_size
                elif rsi_pullback_long and price_above_1d_hma:
                    trend_signal = current_size * 0.8
            
            # Short: bearish trend + 1d confirmation + pullback or crossover
            if trend_1d_bearish and hma_bearish:
                if hma_bear_cross:
                    trend_signal = -current_size
                elif rsi_pullback_short and price_below_1d_hma:
                    trend_signal = -current_size * 0.8
        
        # === NEUTRAL REGIME: REDUCED SIGNALS ===
        neutral_signal = 0.0
        if is_neutral_regime:
            # Only take strong signals in neutral
            if crsi[i] < 10 and close[i] < bb_lower[i]:
                neutral_signal = REDUCED_SIZE
            elif crsi[i] > 90 and close[i] > bb_upper[i]:
                neutral_signal = -REDUCED_SIZE
        
        # === COMBINE SIGNALS ===
        # Priority: trend regime > range regime > neutral
        if is_trend_regime:
            new_signal = trend_signal
        elif is_range_regime:
            new_signal = range_signal
        else:
            new_signal = neutral_signal
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~17 days on 4h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and hma_8[i] > hma_21[i] and rsi_14[i] > 45:
                new_signal = REDUCED_SIZE
            elif trend_1d_bearish and hma_8[i] < hma_21[i] and rsi_14[i] < 55:
                new_signal = -REDUCED_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend strongly reverses
            if position_side > 0 and trend_1d_bearish and hma_1d_slope_aligned[i] < -0.5:
                trend_reversal = True
            # Exit short if 1d trend strongly reverses
            if position_side < 0 and trend_1d_bullish and hma_1d_slope_aligned[i] > 0.5:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals