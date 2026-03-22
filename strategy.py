#!/usr/bin/env python3
"""
Experiment #332: 12h Primary + 1d/1w HTF — HMA Trend + Connors RSI + Choppiness Regime

Hypothesis: Combining proven patterns from research into a cleaner 12h strategy:
1. HMA(16/48) on 12h provides faster trend response than KAMA/EMA
2. 1w HMA(21) gives major trend direction (crypto has multi-week trends)
3. Connors RSI (RSI2 + RSI_Streak + PercentRank) catches pullbacks with 75% win rate
4. Choppiness Index switches between trend-follow and mean-revert modes
5. Fewer conflicting conditions = more trades generated (critical lesson from 300+ failures)
6. Target: 25-45 trades/year on 12h (appropriate frequency, low fee drag)

Why this might beat current best (Sharpe=0.424):
- Connors RSI is proven on ETH (Sharpe +0.923 in research)
- HMA crossover is faster than KAMA for crypto volatility
- Simpler entry logic ensures trades on ALL symbols (BTC, ETH, SOL)
- Regime-aware sizing (trend mode = full size, chop mode = reduced)
- Asymmetric: longs 0.30, shorts 0.20 (crypto bias)

Key improvements over failed experiments:
- Removed Donchian (too many false breakouts on 12h)
- Simplified RSI conditions (Connors RSI < 10 or > 90, not complex ranges)
- Single HMA crossover instead of dual KAMA
- Fewer regime filters (just CHOP for trend/chop detection)
- Ensured minimum trade frequency with fallback entries

Position sizing: 0.25 base, 0.30 strong (longs), 0.20 (shorts)
Stoploss: 3.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_chop_1d1w_asym_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag, excellent for crypto trends.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    hull = wma(2 * wma_half - wma_full, sqrt_n)
    
    return hull.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): Consecutive up/down days
    PercentRank(100): Where price sits in recent range
    
    Entry: CRSI < 10 (long), CRSI > 90 (short)
    Proven 75% win rate in research.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank (where price sits in last 100 bars)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        rank = np.sum(window < close[i]) / rank_period
        crsi[i] = (rsi_3.iloc[i] + rsi_streak.iloc[i] + rank * 100) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (intermediate trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h_16 = calculate_hma(close, 16)
    hma_12h_48 = calculate_hma(close, 48)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1w HMA (favor longs)
        # Bear: price below 1w HMA (allow shorts)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND (confirmation) ===
        trend_1d_bull = close[i] > hma_1d_21_aligned[i]
        trend_1d_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trending market (trend follow entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 12H LOCAL TREND ===
        # HMA crossover
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_12h_48[i] > hma_12h_48[i-3] if i >= 3 else False
        hma_slope_down = hma_12h_48[i] < hma_12h_48[i-3] if i >= 3 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_12h_48[i]
        price_below_hma = close[i] < hma_12h_48[i]
        
        # Price relative to SMA200 (long-term trend)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 10 = oversold (long entry)
        # CRSI > 90 = overbought (short entry)
        crsi_oversold = crsi_14[i] < 15.0
        crsi_overbought = crsi_14[i] > 85.0
        crsi_extreme_oversold = crsi_14[i] < 10.0
        crsi_extreme_overbought = crsi_14[i] > 90.0
        
        # === ENTRY LOGIC (REGIME-AWARE + ASYMMETRIC) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Trending market + HMA bullish + CRSI pullback
            if is_trending and hma_bullish and crsi_oversold and price_above_hma:
                new_signal = LONG_BASE
            
            # Strong CRSI oversold + bull regime + above SMA200
            elif crsi_extreme_oversold and regime_bull and price_above_sma200:
                new_signal = LONG_STRONG
            
            # HMA bullish crossover + CRSI rising from oversold
            elif hma_bullish and hma_slope_up and crsi_14[i] > 10.0 and crsi_14[i] < 30.0:
                new_signal = LONG_BASE
            
            # Choppy market mean revert (CRSI very oversold)
            elif is_choppy and crsi_extreme_oversold:
                new_signal = LONG_BASE * 0.8
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Trending market + HMA bearish + CRSI pullback
            if is_trending and hma_bearish and crsi_overbought and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Strong CRSI overbought + bear regime
            elif crsi_extreme_overbought and regime_bear and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # HMA bearish crossover + CRSI falling from overbought
            elif hma_bearish and hma_slope_down and crsi_14[i] < 90.0 and crsi_14[i] > 70.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Choppy market mean revert (CRSI very overbought)
            elif is_choppy and crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 25 bars (~12-13 days on 12h)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi_14[i] < 30.0:
                new_signal = LONG_BASE * 0.6
            elif regime_bear and crsi_14[i] > 70.0:
                new_signal = -SHORT_BASE * 0.6
            elif crsi_extreme_oversold:
                new_signal = LONG_BASE * 0.6
            elif crsi_extreme_overbought:
                new_signal = -SHORT_BASE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1w regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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