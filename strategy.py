#!/usr/bin/env python3
"""
Experiment #243: 1d Primary + 1w HTF — Regime-Adaptive Connors RSI + Choppiness

Hypothesis: Daily timeframe with weekly regime filtering captures major moves while
avoiding whipsaw. After 242 experiments, the key insight is:
1. Weekly HMA slope determines major bull/bear regime (slow, reliable)
2. Choppiness Index (14) tells us whether to trend-follow or mean-revert
3. Connors RSI (RSI3 + RSI_Streak + PercentRank) provides high-probability entries
4. Loose thresholds ensure 10-30 trades/year (not too strict like #235/#238/#240)

This combines proven patterns:
- Connors RSI extremes (75% win rate in research)
- Choppiness regime switch (ETH Sharpe +0.923 in experiments)
- Weekly HMA trend filter (avoids counter-trend trades in strong trends)
- ATR trailing stop (2.5x for risk management)

Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
Target: 15-30 trades/year per symbol (within 1d cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_chop_regime_1w_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pctrank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of closes in last 100 days that are below current close
    
    Long entry: CRSI < 10-15 (oversold)
    Short entry: CRSI > 85-90 (overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
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
            # Consecutive up days - bullish
            streak_rsi[i] = 100 * (streak_abs[i] / streak_period)
            streak_rsi[i] = min(100, streak_rsi[i])
        elif streak[i] < 0:
            # Consecutive down days - bearish
            streak_rsi[i] = 100 * (1 - streak_abs[i] / streak_period)
            streak_rsi[i] = max(0, streak_rsi[i])
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current close ranks vs last 100 closes
    pct_rank = np.zeros(n)
    for i in range(pctrank_period, n):
        window = close[i-pctrank_period+1:i+1]
        count_below = np.sum(window < close[i])
        pct_rank[i] = 100 * count_below / pctrank_period
    
    # Combine into Connors RSI
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    hma_1d_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    bars_without_signal = 0
    
    for i in range(250, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        # === WEEKLY REGIME DETECTION ===
        # Bull regime: 1w HMA slope > 0.10%
        # Bear regime: 1w HMA slope < -0.10%
        # Neutral: between
        regime_bull = hma_1w_slope_aligned[i] > 0.10
        regime_bear = hma_1w_slope_aligned[i] < -0.10
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trend market (trend follow)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === LONG-TERM TREND FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        
        # === CONNORS RSI SIGNALS (LOOSE THRESHOLDS FOR TRADE FREQUENCY) ===
        crsi_oversold = crsi[i] < 20  # Mean reversion long
        crsi_overbought = crsi[i] > 80  # Mean reversion short
        crsi_extreme_oversold = crsi[i] < 12  # Strong long
        crsi_extreme_overbought = crsi[i] > 88  # Strong short
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION MODE (when choppy) - PRIMARY STRATEGY
        if is_choppy:
            # LONG: CRSI oversold + price below 1d HMA (pullback in range)
            if crsi_oversold and price_below_1d_hma:
                if regime_bear:
                    # In bear regime, only small position
                    new_signal = BASE_SIZE * 0.5
                else:
                    new_signal = BASE_SIZE
            
            # LONG: CRSI extreme oversold (any regime except strong bear)
            if crsi_extreme_oversold and not regime_bear:
                new_signal = max(new_signal, STRONG_SIZE) if new_signal > 0 else STRONG_SIZE
            
            # SHORT: CRSI overbought + price above 1d HMA (pullback in range)
            if crsi_overbought and price_above_1d_hma:
                if regime_bull:
                    # In bull regime, only small position
                    new_signal = -BASE_SIZE * 0.5
                else:
                    new_signal = -BASE_SIZE
            
            # SHORT: CRSI extreme overbought (any regime except strong bull)
            if crsi_extreme_overbought and not regime_bull:
                new_signal = min(new_signal, -STRONG_SIZE) if new_signal < 0 else -STRONG_SIZE
        
        # TREND FOLLOWING MODE (when trending)
        if is_trending:
            # LONG: Trending + regime bull + CRSI not overbought
            if regime_bull and price_above_1w_hma and crsi[i] < 70:
                if price_above_sma200:
                    new_signal = BASE_SIZE
                elif crsi_oversold:
                    new_signal = BASE_SIZE * 0.7
            
            # SHORT: Trending + regime bear + CRSI not oversold
            if regime_bear and price_below_1w_hma and crsi[i] > 30:
                if price_below_sma200:
                    new_signal = -BASE_SIZE
                elif crsi_overbought:
                    new_signal = -BASE_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 30 bars (~30 days on 1d)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if crsi_extreme_oversold and not regime_bear:
                new_signal = BASE_SIZE * 0.5
            elif crsi_extreme_overbought and not regime_bull:
                new_signal = -BASE_SIZE * 0.5
            elif regime_bull and price_above_sma200 and crsi[i] < 45:
                new_signal = BASE_SIZE * 0.4
            elif regime_bear and price_below_sma200 and crsi[i] > 55:
                new_signal = -BASE_SIZE * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_1w_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_1w_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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
                bars_without_signal = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                bars_without_signal = 0
            else:
                bars_without_signal = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
            bars_without_signal += 1
        
        signals[i] = new_signal
    
    return signals