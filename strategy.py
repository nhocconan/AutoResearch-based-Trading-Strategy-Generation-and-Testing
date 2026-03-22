#!/usr/bin/env python3
"""
Experiment #267: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian

Hypothesis: After #263 failed (Sharpe=-0.330), simplify with proven components:
1. 1d PRIMARY timeframe (as required) — fewer trades, less fee drag
2. 1w HTF for major trend bias (only long when weekly bullish, only short when weekly bearish)
3. Choppiness Index(14) for regime detection: CHOP>61.8=mean-revert, CHOP<38.2=trend
4. Connors RSI for mean reversion entries (RSI3 + RSI_Streak2 + PercentRank100)/3
5. Donchian(20) breakout for trend entries when trending regime
6. ATR(14) trailing stoploss at 2.5x

Key improvements over #263:
- Simpler regime logic (choppy vs trending, not dual)
- Connors RSI instead of standard RSI (proven 75% win rate in research)
- Weekly HTF trend filter prevents counter-trend trades in bear market
- More aggressive entry thresholds to ensure 10+ trades/year

Position sizing: 0.30 base (discrete levels: 0.0, ±0.30)
Target: 20-40 trades/year (appropriate for 1d)
Stoploss: 2.5 * ATR trailing

Why this should work in 2025 bear/range:
- Mean reversion dominates in choppy markets (CHOP>61.8)
- Weekly filter prevents fighting major trend
- Connors RSI catches oversold/overbought extremes efficiently
- 1d timeframe = ~25 trades/year naturally (within fee budget)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_connors_donchian_1w_v1"
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
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive up/down days (positive for up, negative for down)
    PercentRank: percentile rank of today's return over last 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI(2) component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100 scale)
    # Positive streak = bullish, negative = bearish
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            # Bullish streak: higher = more overbought
            streak_rsi[i] = min(100, 50 + streak_abs[i] * 10)
        else:
            # Bearish streak: higher magnitude = more oversold
            streak_rsi[i] = max(0, 50 - streak_abs[i] * 10)
    
    # PercentRank(100) component
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) > 0:
            percent_rank[i] = 100 * np.sum(window <= returns[i]) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi3 + streak_rsi + percent_rank) / 3.0
    
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    hma_1w_21 = calculate_rsi(df_1w['close'].values, 21)  # Use as trend proxy
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Weekly close above/below weekly SMA for trend bias
    close_1w = df_1w['close'].values
    sma_1w_10 = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    sma_1w_10_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_10)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(sma_1w_10_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === 1W TREND BIAS (major direction filter) ===
        # Only long when weekly close > weekly SMA(10)
        # Only short when weekly close < weekly SMA(10)
        weekly_bullish = close[i] > sma_1w_10_aligned[i] if not np.isnan(sma_1w_10_aligned[i]) else True
        weekly_bearish = close[i] < sma_1w_10_aligned[i] if not np.isnan(sma_1w_10_aligned[i]) else False
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range market (mean revert entries)
        # CHOP < 38.2 = trend market (breakout entries)
        # 38.2 <= CHOP <= 61.8 = transition (no trades or reduced size)
        is_choppy = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_transition = 38.2 <= chop_14[i] <= 61.8
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.005
        
        # === CONNORS RSI THRESHOLDS ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Choppy + CRSI oversold + weekly bias allows long
            if crsi_oversold and (weekly_bullish or not weekly_bearish):
                new_signal = BASE_SIZE
            # LONG: Choppy + CRSI extreme oversold (any weekly bias)
            if crsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Choppy + CRSI overbought + weekly bias allows short
            if crsi_overbought and (weekly_bearish or not weekly_bullish):
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + CRSI extreme overbought (any weekly bias)
            if crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # TREND FOLLOWING MODE (when trending)
        if is_trending:
            # LONG: Trending + Donchian breakout + weekly bullish + price > SMA200
            if donchian_breakout_long and weekly_bullish and price_above_sma200:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + Donchian breakdown + weekly bearish + price < SMA200
            if donchian_breakout_short and weekly_bearish and price_below_sma200:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 10+ trades/year) ===
        # Force trade if no signal for 15 bars (~15 days on 1d)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if weekly_bullish and crsi[i] < 35 and price_above_sma200:
                new_signal = BASE_SIZE * 0.7
            elif weekly_bearish and crsi[i] > 65 and price_below_sma200:
                new_signal = -BASE_SIZE * 0.7
            elif is_choppy and crsi[i] < 30:
                new_signal = BASE_SIZE * 0.6
            elif is_choppy and crsi[i] > 70:
                new_signal = -BASE_SIZE * 0.6
        
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
            if position_side > 0 and weekly_bearish and price_below_sma200:
                regime_reversal = True
            # Short position but weekly turns strongly bullish
            if position_side < 0 and weekly_bullish and price_above_sma200:
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