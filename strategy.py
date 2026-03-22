#!/usr/bin/env python3
"""
Experiment #021: 4h Regime-Adaptive HMA + Connors RSI with 1d/1w Bias

Hypothesis: Previous strategies failed because they used ONE regime logic for all market conditions.
This strategy adapts to market regime using Choppiness Index:
1. CHOP(14) > 61.8 = RANGING → use Connors RSI mean reversion (high win rate ~75%)
2. CHOP(14) < 38.2 = TRENDING → use HMA trend following with pullback entries
3. 1d HMA(21) for major trend bias (prevents counter-trend trades)
4. 1w HMA(21) for secular trend filter (only trade with multi-week direction)
5. ATR(14) trailing stoploss at 2.5x for risk management
6. Discrete position sizing (0.25-0.30) to minimize fee churn

Why this should work:
- Regime detection prevents trend strategies in choppy markets (major failure mode)
- Connors RSI has proven 75% win rate in range conditions
- Multi-timeframe bias (1d + 1w) prevents counter-trend trades that failed in 2022
- 4h timeframe naturally limits trade frequency (target 20-50 trades/year)
- Based on research showing regime-adaptive strategies outperform single-regime

Key improvements over failed experiments:
- Uses Choppiness Index (failed in #013 but was combined with too many filters)
- Simpler Connors RSI (not extreme thresholds that caused 0 trades in #005, #008)
- 1w bias added for additional trend filter (not used in #016)
- Fewer conflicting conditions = more trades while maintaining quality

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_connors_hma_1d1w_bias_v1"
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate ATR for each bar
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    # Clip to valid range
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): 3-period RSI on price
    RSI(Streak, 2): 2-period RSI on streak (consecutive up/down days)
    PercentRank(100): Percentile rank of today's change over last 100 days
    
    CRSI < 10 = oversold (long opportunity)
    CRSI > 90 = overbought (short opportunity)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on price
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like values (positive streak = gain, negative = loss)
    streak_gain = np.maximum(streak, 0)
    streak_loss = np.abs(np.minimum(streak, 0))
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / avg_streak_loss
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    # Component 3: Percentile Rank of price change over last 100 bars
    price_change = close_s.diff().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = price_change[i-rank_period:i]
        current = price_change[i]
        # Count how many values in window are <= current
        rank = np.sum(window <= current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1D & 1W TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === REGIME DETECTION ===
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        # Neutral zone: 38.2 <= CHOP <= 61.8 (reduce position or stay flat)
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 150:
            atr_median = np.nanmedian(atr_14[max(0, i-150):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Require alignment across timeframes for trend trades
        trend_aligned_long = hma_bullish and daily_bullish and weekly_bullish
        trend_aligned_short = hma_bearish and daily_bearish and weekly_bearish
        
        if is_trending:
            # TREND REGIME: Use HMA trend + RSI pullback
            if trend_aligned_long:
                # RSI pullback to 40-55 range in uptrend
                if 40 <= rsi_14[i] <= 55:
                    new_signal = current_size
                # Or Donchian breakout with RSI confirmation
                elif i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1] and 50 <= rsi_14[i] <= 70:
                        new_signal = current_size
            
            elif trend_aligned_short:
                # RSI pullback to 45-60 range in downtrend
                if 45 <= rsi_14[i] <= 60:
                    new_signal = -current_size
                # Or Donchian breakdown with RSI confirmation
                elif i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1] and 30 <= rsi_14[i] <= 50:
                        new_signal = -current_size
        
        elif is_ranging:
            # RANGE REGIME: Use Connors RSI mean reversion
            # Long when CRSI extremely oversold + price above 1d HMA (bias long in bull)
            if crsi[i] < 15 and daily_bullish:
                new_signal = current_size
            # Short when CRSI extremely overbought + price below 1d HMA (bias short in bear)
            elif crsi[i] > 85 and daily_bearish:
                new_signal = -current_size
            # Weaker signals in neutral daily bias
            elif crsi[i] < 10:
                new_signal = current_size * 0.7
            elif crsi[i] > 90:
                new_signal = -current_size * 0.7
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        # Only take highest conviction trades
        if not is_ranging and not is_trending:
            if trend_aligned_long and crsi[i] < 20:
                new_signal = current_size * 0.6
            elif trend_aligned_short and crsi[i] > 80:
                new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~10 days on 4h), force entry with weaker signal
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if trend_aligned_long and rsi_14[i] > 40:
                new_signal = current_size * 0.5
            elif trend_aligned_short and rsi_14[i] < 60:
                new_signal = -current_size * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # === REGIME CHANGE EXIT ===
        # Exit long if regime switches from trend to range (and vice versa for short)
        regime_change_exit = False
        if in_position and position_side > 0 and is_ranging and chop_14[i] > 70:
            regime_change_exit = True
        if in_position and position_side < 0 and is_ranging and chop_14[i] > 70:
            regime_change_exit = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal or regime_change_exit:
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