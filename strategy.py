#!/usr/bin/env python3
"""
Experiment #232: 12h Primary + 1d/1w HTF — Regime-Adaptive Connors RSI + Choppiness Filter

Hypothesis: After 231 experiments, the key insight is REGIME DETECTION + ADAPTIVE LOGIC.
1. 12h PRIMARY timeframe = 20-50 trades/year target (matches cost model perfectly)
2. CHOPPINESS INDEX (14) to detect range vs trend regime
3. CONNORS RSI for mean-reversion entries in choppy markets (75% win rate in ranges)
4. HMA(21) slope for trend-following entries in trending markets
5. 1d/1w HTF for major bias filter (avoid fighting macro trend)
6. ATR(14) 2.5x trailing stop for risk management

Why this should work on 12h:
- CHOP > 61.8 = range regime → use Connors RSI mean reversion (buy <10, sell >90)
- CHOP < 38.2 = trend regime → use HMA slope + Donchian breakout
- 12h timeframe naturally filters noise, fewer false signals than 4h/1h
- LOOSE Connors thresholds (12/88 not 10/90) ensure trade frequency
- HTF alignment prevents fighting major trends (1w bias is critical)

Key differences from failed #226 (same TF but different logic):
- Dual-regime switching (not just Connors always)
- Added Donchian breakout for trend entries
- Added 1w HTF for macro bias (was only 1d before)
- Force-trade fallback after 40 bars of no signal
- Smaller position sizes (0.25-0.30) for lower DD

Position sizing: 0.25 base, 0.30 strong signals
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_connors_chop_hma_1d1w_v1"
timeframe = "12h"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    price_range = hh - ll
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100 scale)
    streak_positive = np.maximum(streak, 0)
    streak_negative = np.abs(np.minimum(streak, 0))
    
    avg_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    rs_streak = avg_gain / avg_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: Percentile Rank (where current close ranks vs last 100 closes)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine all three components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    adx_12h = calculate_adx(high, low, close, 14)
    
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
    last_trade_bar = -40
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range/mean-reversion regime
        is_trending = chop[i] < 45.0  # Trend-following regime
        # 45-55 = neutral, use conservative signals
        
        # === HTF TREND BIAS (1w) ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.05
        weekly_bearish = hma_1w_slope_aligned[i] < -0.05
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === INTERMEDIATE TREND (1d) ===
        daily_bullish = hma_1d_slope_aligned[i] > 0.10
        daily_bearish = hma_1d_slope_aligned[i] < -0.10
        daily_trend_strength = adx_1d_aligned[i] > 25
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === LOCAL TREND (12h HMA) ===
        local_bullish = hma_12h_slope[i] > 0.15
        local_bearish = hma_12h_slope[i] < -0.15
        
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        
        # === MOMENTUM (RSI + Connors RSI) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        crsi_oversold = crsi[i] < 15  # Loose threshold for trade frequency
        crsi_overbought = crsi[i] > 85  # Loose threshold
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === LOCAL TREND STRENGTH ===
        local_trend_strong = adx_12h[i] > 25
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_score = 0
        long_strength = 0
        
        if is_choppy:
            # MEAN REVERSION MODE (Choppy market)
            # Path 1: Connors RSI extreme oversold + weekly/daily neutral-bullish
            if crsi_oversold and (weekly_bullish or weekly_neutral):
                long_score += 4
                long_strength = STRONG_SIZE
            
            # Path 2: Connors RSI oversold + RSI oversold (double confirmation)
            if crsi_oversold and rsi_oversold:
                long_score += 4
                long_strength = STRONG_SIZE
            
            # Path 3: Connors RSI oversold + price below 12h HMA (pullback)
            if crsi_oversold and price_below_12h_hma and price_above_1d_hma:
                long_score += 3
                long_strength = BASE_SIZE
            
            # Path 4: RSI oversold + weekly bullish (simple)
            if rsi_oversold and weekly_bullish and bars_since_last_trade > 20:
                long_score += 3
                long_strength = BASE_SIZE
            
            # Path 5: Connors RSI very oversold (<10)
            if crsi[i] < 10 and bars_since_last_trade > 15:
                long_score += 3
                long_strength = BASE_SIZE
        
        if is_trending:
            # TREND FOLLOWING MODE (Trending market)
            # Path 1: Donchian breakout + weekly bullish + local bullish
            if breakout_long and weekly_bullish and local_bullish:
                long_score += 5
                long_strength = STRONG_SIZE
            
            # Path 2: Donchian breakout + daily bullish
            if breakout_long and daily_bullish:
                long_score += 4
                long_strength = STRONG_SIZE
            
            # Path 3: HMA slope bullish + price above HMA + ADX strong
            if local_bullish and price_above_12h_hma and local_trend_strong:
                long_score += 3
                long_strength = BASE_SIZE
            
            # Path 4: Breakout + daily bullish + RSI confirmation
            if breakout_long and daily_bullish and rsi_14[i] > 50:
                long_score += 3
                long_strength = BASE_SIZE
            
            # Path 5: All HTF aligned bullish + local breakout
            if weekly_bullish and daily_bullish and breakout_long:
                long_score += 4
                long_strength = STRONG_SIZE
        
        # Neutral regime (45-55 chop) - use conservative signals
        if not is_choppy and not is_trending:
            if weekly_bullish and daily_bullish and price_above_12h_hma and rsi_14[i] > 45:
                long_score += 2
                long_strength = BASE_SIZE * 0.7
            
            if breakout_long and weekly_bullish:
                long_score += 2
                long_strength = BASE_SIZE * 0.7
        
        # Apply score thresholds
        if long_score >= 4:
            new_signal = long_strength
        elif long_score >= 3 and bars_since_last_trade > 15:
            new_signal = long_strength * 0.9
        elif long_score >= 2 and bars_since_last_trade > 30:
            new_signal = long_strength * 0.7
        
        # SHORT ENTRIES
        short_score = 0
        short_strength = 0
        
        if is_choppy:
            # MEAN REVERSION MODE (Choppy market)
            # Path 1: Connors RSI extreme overbought + weekly/daily neutral-bearish
            if crsi_overbought and (weekly_bearish or weekly_neutral):
                short_score += 4
                short_strength = STRONG_SIZE
            
            # Path 2: Connors RSI overbought + RSI overbought (double confirmation)
            if crsi_overbought and rsi_overbought:
                short_score += 4
                short_strength = STRONG_SIZE
            
            # Path 3: Connors RSI overbought + price above 12h HMA (pullback)
            if crsi_overbought and price_above_12h_hma and price_below_1d_hma:
                short_score += 3
                short_strength = BASE_SIZE
            
            # Path 4: RSI overbought + weekly bearish (simple)
            if rsi_overbought and weekly_bearish and bars_since_last_trade > 20:
                short_score += 3
                short_strength = BASE_SIZE
            
            # Path 5: Connors RSI very overbought (>90)
            if crsi[i] > 90 and bars_since_last_trade > 15:
                short_score += 3
                short_strength = BASE_SIZE
        
        if is_trending:
            # TREND FOLLOWING MODE (Trending market)
            # Path 1: Donchian breakout + weekly bearish + local bearish
            if breakout_short and weekly_bearish and local_bearish:
                short_score += 5
                short_strength = STRONG_SIZE
            
            # Path 2: Donchian breakout + daily bearish
            if breakout_short and daily_bearish:
                short_score += 4
                short_strength = STRONG_SIZE
            
            # Path 3: HMA slope bearish + price below HMA + ADX strong
            if local_bearish and price_below_12h_hma and local_trend_strong:
                short_score += 3
                short_strength = BASE_SIZE
            
            # Path 4: Breakout + daily bearish + RSI confirmation
            if breakout_short and daily_bearish and rsi_14[i] < 50:
                short_score += 3
                short_strength = BASE_SIZE
            
            # Path 5: All HTF aligned bearish + local breakout
            if weekly_bearish and daily_bearish and breakout_short:
                short_score += 4
                short_strength = STRONG_SIZE
        
        # Neutral regime (45-55 chop) - use conservative signals
        if not is_choppy and not is_trending:
            if weekly_bearish and daily_bearish and price_below_12h_hma and rsi_14[i] < 55:
                short_score += 2
                short_strength = BASE_SIZE * 0.7
            
            if breakout_short and weekly_bearish:
                short_score += 2
                short_strength = BASE_SIZE * 0.7
        
        # Apply score thresholds
        if short_score >= 4:
            new_signal = -short_strength
        elif short_score >= 3 and bars_since_last_trade > 15:
            new_signal = -short_strength * 0.9
        elif short_score >= 2 and bars_since_last_trade > 30:
            new_signal = -short_strength * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 40 bars (~20 days on 12h)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if weekly_bullish and crsi[i] < 25 and price_above_1d_hma:
                new_signal = BASE_SIZE * 0.35
            elif weekly_bearish and crsi[i] > 75 and price_below_1d_hma:
                new_signal = -BASE_SIZE * 0.35
            elif daily_bullish and local_bullish and rsi_14[i] > 48:
                new_signal = BASE_SIZE * 0.30
            elif daily_bearish and local_bearish and rsi_14[i] < 52:
                new_signal = -BASE_SIZE * 0.30
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but weekly turns strongly bearish
            if position_side > 0 and weekly_bearish and price_below_1w_hma:
                trend_reversal = True
            # Short position but weekly turns strongly bullish
            if position_side < 0 and weekly_bullish and price_above_1w_hma:
                trend_reversal = True
        
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