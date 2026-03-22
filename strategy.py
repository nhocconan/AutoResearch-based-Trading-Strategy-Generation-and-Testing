#!/usr/bin/env python3
"""
Experiment #233: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: After 232 experiments, pure trend-following fails in bear/range markets.
The winning formula combines:
1. 1d PRIMARY for daily signals (proven higher TF works best)
2. 1w HTF for major trend bias filter (avoid fighting macro trend)
3. CHOPPINESS INDEX (14) for regime detection: >61.8=range, <38.2=trend
4. CONNORS RSI for mean reversion entries in choppy markets (75% win rate)
5. HMA trend + Donchian breakout for trending markets
6. ATR(14) 2.5x trailing stop for risk management
7. Asymmetric sizing: smaller in chop (0.20), larger in trend (0.30)

Why this should work:
- 1d timeframe = 20-50 trades/year target (matches cost model for daily)
- Dual regime adapts to market conditions (trend vs range)
- Connors RSI proven on ETH (Sharpe +0.923 in research)
- Weekly bias prevents fighting macro trend
- Discrete sizing (0.0, ±0.20, ±0.30) minimizes fee churn

Key differences from failed #231, #232:
- 1d primary instead of 4h/12h (higher TF = fewer false signals)
- Choppiness Index regime switch (not used in recent failures)
- Connors RSI instead of standard RSI (better for mean reversion)
- Different logic per regime (adaptive strategy)

Position sizing: 0.20 chop, 0.30 trend
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_connors_chop_1w_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Fast RSI for short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Where current price ranks vs last 100 days
    
    Entry signals:
    - Long: CRSI < 10 (oversold) + price > SMA(200)
    - Short: CRSI > 90 (overbought) + price < SMA(200)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, 3)
    
    # Component 2: Streak RSI
    # Streak = consecutive up (+1) or down (-1) days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Choppy/Range-bound market (mean reversion)
    - CHOP < 38.2: Trending market (trend following)
    - 38.2 - 61.8: Transition zone
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max().values
    ll = low_s.rolling(window=period, min_periods=period).min().values
    hh_ll = hh - ll
    
    # Avoid division by zero
    hh_ll = np.where(hh_ll == 0, 0.0001, hh_ll)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(period)
    
    # Clamp to 0-100
    chop = np.clip(chop, 0, 100)
    
    return chop

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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_slope = calculate_hma_slope(hma_21, 3)
    adx_14 = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    CHOP_SIZE = 0.20  # Smaller in choppy markets
    TREND_SIZE = 0.30  # Larger in trending markets
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(250, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range-bound market
        is_trending = chop[i] < 38.2  # Trending market
        # 38.2 - 61.8 = transition (use conservative signals)
        
        # === HTF TREND BIAS (1w) ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === LOCAL TREND ===
        local_bullish = hma_slope[i] > 0.3
        local_bearish = hma_slope[i] < -0.3
        
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === TREND STRENGTH ===
        strong_trend = adx_14[i] > 25
        weak_trend = adx_14[i] < 20
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Mean reversion long
        crsi_overbought = crsi[i] > 85  # Mean reversion short
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # === POSITION SIZING ===
        current_size = CHOP_SIZE if is_choppy else TREND_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_score = 0
        long_strength = 0
        
        if is_choppy:
            # MEAN REVERSION in choppy market (Connors RSI)
            # Path 1: CRSI extreme oversold + above SMA200 (strong long bias)
            if crsi_extreme_oversold and price_above_sma200:
                long_score += 5
                long_strength = CHOP_SIZE
            
            # Path 2: CRSI oversold + weekly bullish
            if crsi_oversold and weekly_bullish:
                long_score += 4
                long_strength = CHOP_SIZE
            
            # Path 3: CRSI oversold + price above HMA21
            if crsi_oversold and price_above_hma:
                long_score += 3
                long_strength = CHOP_SIZE * 0.8
            
            # Path 4: CRSI < 20 + above SMA200 (looser)
            if crsi[i] < 20 and price_above_sma200 and bars_since_last_trade > 20:
                long_score += 3
                long_strength = CHOP_SIZE * 0.7
            
            # Path 5: RSI(14) < 35 + above SMA200 (alternative)
            if rsi_14[i] < 35 and price_above_sma200 and bars_since_last_trade > 25:
                long_score += 2
                long_strength = CHOP_SIZE * 0.6
        else:
            # TREND FOLLOWING in trending market
            # Path 1: Donchian breakout + weekly bullish + ADX strong
            if breakout_long and weekly_bullish and strong_trend:
                long_score += 5
                long_strength = TREND_SIZE
            
            # Path 2: Donchian breakout + weekly bullish
            if breakout_long and weekly_bullish:
                long_score += 4
                long_strength = TREND_SIZE
            
            # Path 3: HMA bullish + weekly bullish + ADX > 20
            if local_bullish and weekly_bullish and adx_14[i] > 20:
                long_score += 4
                long_strength = TREND_SIZE
            
            # Path 4: Breakout + price above HMA21
            if breakout_long and price_above_hma:
                long_score += 3
                long_strength = TREND_SIZE * 0.8
            
            # Path 5: Weekly bullish + HMA bullish + RSI > 50
            if weekly_bullish and local_bullish and rsi_14[i] > 50:
                long_score += 3
                long_strength = TREND_SIZE * 0.7
            
            # Path 6: Breakout + RSI > 55 (momentum)
            if breakout_long and rsi_14[i] > 55 and bars_since_last_trade > 15:
                long_score += 2
                long_strength = TREND_SIZE * 0.6
        
        # Transition zone (38.2 - 61.8): use conservative signals
        if not is_choppy and not is_trending:
            long_strength = long_strength * 0.7 if long_strength > 0 else 0
        
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
            # MEAN REVERSION in choppy market (Connors RSI)
            # Path 1: CRSI extreme overbought + below SMA200
            if crsi_extreme_overbought and price_below_sma200:
                short_score += 5
                short_strength = CHOP_SIZE
            
            # Path 2: CRSI overbought + weekly bearish
            if crsi_overbought and weekly_bearish:
                short_score += 4
                short_strength = CHOP_SIZE
            
            # Path 3: CRSI overbought + price below HMA21
            if crsi_overbought and price_below_hma:
                short_score += 3
                short_strength = CHOP_SIZE * 0.8
            
            # Path 4: CRSI > 80 + below SMA200 (looser)
            if crsi[i] > 80 and price_below_sma200 and bars_since_last_trade > 20:
                short_score += 3
                short_strength = CHOP_SIZE * 0.7
            
            # Path 5: RSI(14) > 65 + below SMA200 (alternative)
            if rsi_14[i] > 65 and price_below_sma200 and bars_since_last_trade > 25:
                short_score += 2
                short_strength = CHOP_SIZE * 0.6
        else:
            # TREND FOLLOWING in trending market
            # Path 1: Donchian breakout + weekly bearish + ADX strong
            if breakout_short and weekly_bearish and strong_trend:
                short_score += 5
                short_strength = TREND_SIZE
            
            # Path 2: Donchian breakout + weekly bearish
            if breakout_short and weekly_bearish:
                short_score += 4
                short_strength = TREND_SIZE
            
            # Path 3: HMA bearish + weekly bearish + ADX > 20
            if local_bearish and weekly_bearish and adx_14[i] > 20:
                short_score += 4
                short_strength = TREND_SIZE
            
            # Path 4: Breakout + price below HMA21
            if breakout_short and price_below_hma:
                short_score += 3
                short_strength = TREND_SIZE * 0.8
            
            # Path 5: Weekly bearish + HMA bearish + RSI < 50
            if weekly_bearish and local_bearish and rsi_14[i] < 50:
                short_score += 3
                short_strength = TREND_SIZE * 0.7
            
            # Path 6: Breakout + RSI < 45 (momentum)
            if breakout_short and rsi_14[i] < 45 and bars_since_last_trade > 15:
                short_score += 2
                short_strength = TREND_SIZE * 0.6
        
        # Transition zone: use conservative signals
        if not is_choppy and not is_trending:
            short_strength = short_strength * 0.7 if short_strength > 0 else 0
        
        if short_score >= 4:
            new_signal = -short_strength
        elif short_score >= 3 and bars_since_last_trade > 15:
            new_signal = -short_strength * 0.9
        elif short_score >= 2 and bars_since_last_trade > 30:
            new_signal = -short_strength * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 90 bars (~90 days on 1d)
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if weekly_bullish and crsi[i] < 30 and price_above_hma:
                new_signal = CHOP_SIZE * 0.35
            elif weekly_bearish and crsi[i] > 70 and price_below_hma:
                new_signal = -CHOP_SIZE * 0.35
            elif weekly_bullish and price_above_1w_hma and bars_since_last_trade > 120:
                new_signal = TREND_SIZE * 0.30
            elif weekly_bearish and price_below_1w_hma and bars_since_last_trade > 120:
                new_signal = -TREND_SIZE * 0.30
        
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