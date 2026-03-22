#!/usr/bin/env python3
"""
Experiment #237: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: After 236 experiments, the clearest pattern is that simple trend-following
fails on BTC/ETH in bear/range markets (2025+). Connors RSI has documented 75% win rate
for mean reversion. Combined with 1w HMA for major trend bias and Choppiness Index for
regime detection, this should work across all market conditions.

Key components:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Entry long: CRSI < 15 (extreme oversold)
   - Entry short: CRSI > 85 (extreme overbought)
2. 1w HMA(21) slope for major trend bias (only trade with weekly trend)
3. Choppiness Index(14) to avoid trading in extreme chop (CHOP > 70)
4. 1d ATR(14) for 2.5x trailing stop
5. Position size: 0.25 base, 0.30 strong signals (discrete levels)

Why this should work:
- Connors RSI is proven mean-reversion strategy (Larry Connors)
- 1w HTF prevents counter-trend trades in strong trends
- CHOP filter avoids whipsaw in extreme range markets
- 1d timeframe = low trade frequency = low fee drag (target 20-40 trades/year)
- Conservative sizing (0.25-0.30) protects against 2022-style crashes

Position sizing: 0.25 base, 0.30 strong signals
Target: 20-40 trades/year per symbol (within 1d cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_rsi_regime_1w_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on up/down streak length
    PercentRank: Percentile rank of current price over lookback
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on price
    rsi_price = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to positive values for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: Percentile Rank over lookback
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i+1]
        current = close[i]
        count_below = np.sum(lookback < current)
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine components
    crsi = (rsi_price + rsi_streak + percent_rank) / 3.0
    
    # Handle NaN values
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

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
    
    # Calculate 1w HTF indicators (major trend bias)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
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
    last_trade_bar = -50
    
    for i in range(250, n):  # Start after 250 bars for SMA200 + indicators
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === 1W MAJOR TREND BIAS ===
        # Bullish: 1w HMA slope > 0.3%
        # Bearish: 1w HMA slope < -0.3%
        # Neutral: between -0.3% and 0.3%
        weekly_bullish = hma_1w_slope_aligned[i] > 0.30
        weekly_bearish = hma_1w_slope_aligned[i] < -0.30
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 65 = choppy (favor mean reversion)
        # CHOP < 40 = trending (favor trend following)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI SIGNALS (LOOSE THRESHOLDS FOR TRADE FREQUENCY) ===
        # CRSI < 20 = extreme oversold (long signal)
        # CRSI > 80 = extreme overbought (short signal)
        crsi_oversold = crsi[i] < 22  # Slightly higher threshold for more trades
        crsi_overbought = crsi[i] > 78  # Slightly lower threshold for more trades
        
        # Extreme levels for strong signals
        crsi_extreme_oversold = crsi[i] < 15
        crsi_extreme_overbought = crsi[i] > 85
        
        # === POSITION SIZING ===
        new_signal = 0.0
        
        # === ENTRY LOGIC ===
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Strong long: Extreme CRSI oversold + weekly bullish bias
        if crsi_extreme_oversold and (weekly_bullish or weekly_neutral):
            new_signal = STRONG_SIZE
        
        # Standard long: CRSI oversold + price above SMA200 + not weekly bearish
        elif crsi_oversold and price_above_sma200 and not weekly_bearish:
            new_signal = BASE_SIZE
        
        # Mean reversion long in choppy market: CRSI oversold + choppy
        elif crsi_oversold and is_choppy and not weekly_bearish:
            new_signal = BASE_SIZE * 0.8
        
        # SHORT ENTRIES
        # Strong short: Extreme CRSI overbought + weekly bearish bias
        if crsi_extreme_overbought and (weekly_bearish or weekly_neutral):
            new_signal = -STRONG_SIZE
        
        # Standard short: CRSI overbought + price below SMA200 + not weekly bullish
        elif crsi_overbought and price_below_sma200 and not weekly_bullish:
            new_signal = -BASE_SIZE
        
        # Mean reversion short in choppy market: CRSI overbought + choppy
        elif crsi_overbought and is_choppy and not weekly_bullish:
            new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 50 bars (~50 days on 1d)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if crsi_oversold and not weekly_bearish:
                new_signal = BASE_SIZE * 0.5
            elif crsi_overbought and not weekly_bullish:
                new_signal = -BASE_SIZE * 0.5
        
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
            # Long position but weekly turns strongly bearish
            if position_side > 0 and weekly_bearish and price_below_1w_hma:
                regime_reversal = True
            # Short position but weekly turns strongly bullish
            if position_side < 0 and weekly_bullish and price_above_1w_hma:
                regime_reversal = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position but CRSI becomes overbought (>70)
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Short position but CRSI becomes oversold (<30)
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        if stoploss_triggered or regime_reversal or crsi_exit:
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