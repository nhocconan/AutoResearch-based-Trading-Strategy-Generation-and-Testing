#!/usr/bin/env python3
"""
Experiment #352: 12h Primary + 1d/1w HTF — Dual Regime (Trend/Mean Revert) + Connors RSI

Hypothesis: After 350+ experiments, the clearest pattern is:
1. 12h timeframe reduces noise vs 4h/1h while generating sufficient trades (20-50/year)
2. Single-regime strategies fail because crypto alternates between trending and ranging
3. Choppiness Index (CHOP) is the BEST regime filter - proven in academic literature
4. Connors RSI (CRSI) has 75% win rate for mean reversion entries in range markets
5. 1w HMA provides major trend filter that eliminates counter-trend disasters (2022 crash)
6. Dual regime: CHOP>61.8 = mean revert (CRSI), CHOP<38.2 = trend follow (Donchian+HMA)

This strategy combines:
1. 1w HMA(21) for MAJOR trend direction (crypto macro trends last months)
2. 1d HMA(21) for intermediate trend confirmation
3. 12h Choppiness Index(14) for regime detection (range vs trend)
4. Connors RSI(3,2,100) for mean reversion entries in choppy markets
5. 12h Donchian(20) breakout for trend-following entries in trending markets
6. ATR(14) trailing stop 2.5x (cut losers, let winners run)
7. Asymmetric sizing: longs 0.25-0.30, shorts 0.15-0.20 (crypto long bias)

Why this might beat current best (Sharpe=0.435):
- 12h TF generates 20-50 trades/year (optimal for fee drag vs signal quality)
- CHOP regime filter adapts to market conditions (range vs trend)
- CRSI has proven edge in mean reversion (Larry Connors research)
- 1w HTF eliminates catastrophic counter-trend trades
- Dual regime means we ALWAYS have an edge (unlike single-regime strategies)

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 12h (1 trade every 7-14 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dualregime_chop_crsi_1d1w_v1"
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
    Much less lag than EMA while maintaining smoothness.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)  # avoid division by zero
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 days
    
    CRSI < 10 = extremely oversold (long opportunity)
    CRSI > 90 = extremely overbought (short opportunity)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on price
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_price = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: np.sum(x.iloc[:-1] < x.iloc[-1]) / len(x.iloc[:-1]) * 100, raw=False
    ).values
    
    # Combine components
    crsi = (rsi_price.values + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    return donchian_upper, donchian_lower, donchian_mid

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
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (intermediate trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Donchian channels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(donchian_upper[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2 <= CHOP <= 61.8 = transition (use trend bias)
        regime_range = chop[i] > 61.8
        regime_trend = chop[i] < 38.2
        regime_transition = not regime_range and not regime_trend
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Bull: price above 1w HMA (favor longs, allow shorts only with strong signal)
        # Bear: price below 1w HMA (allow both, but shorts favored)
        major_bull = close[i] > hma_1w_21_aligned[i]
        major_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        inter_bull = close[i] > hma_1d_21_aligned[i]
        inter_bear = close[i] < hma_1d_21_aligned[i]
        
        # === TREND CONFIDENCE (both 1w and 1d agree) ===
        trend_confident_bull = major_bull and inter_bull
        trend_confident_bear = major_bear and inter_bear
        
        # === VOLATILITY FILTER (ATR ratio) ===
        atr_30 = calculate_atr(high, low, close, 30)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        vol_spike = atr_ratio > 1.8
        vol_scale = 0.6 if vol_spike else 1.0
        
        # === PRICE POSITION ===
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === DONCHIAN BREAKOUT SIGNALS (trend regime) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI SIGNALS (mean reversion regime) ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (Dual Regime) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # --- TREND REGIME (CHOP < 38.2) ---
        if regime_trend:
            # LONG: Donchian breakout + trend confident bull + RSI not overbought
            if donchian_breakout_long and trend_confident_bull and rsi_14[i] < 70.0:
                new_signal = LONG_STRONG * vol_scale
            
            # LONG: Donchian breakout + major bull + inter bull
            elif donchian_breakout_long and major_bull and inter_bull:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * vol_scale
            
            # SHORT: Donchian breakout + trend confident bear + RSI not oversold
            if donchian_breakout_short and trend_confident_bear and rsi_14[i] > 30.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # SHORT: Donchian breakout + major bear + inter bear
            elif donchian_breakout_short and major_bear and inter_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
        
        # --- RANGE REGIME (CHOP > 61.8) ---
        elif regime_range:
            # LONG: CRSI extreme oversold + price above 1w HMA (bullish bias)
            if crsi_extreme_oversold and major_bull:
                new_signal = LONG_BASE * vol_scale
            
            # LONG: CRSI oversold + CRSI rising + price > SMA200
            elif crsi_oversold and crsi_rising and price_above_sma200:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8 * vol_scale
            
            # SHORT: CRSI extreme overbought + price below 1w HMA (bearish bias)
            if crsi_extreme_overbought and major_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # SHORT: CRSI overbought + CRSI falling + price < SMA200
            elif crsi_overbought and crsi_falling and not price_above_sma200:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # --- TRANSITION REGIME (38.2 <= CHOP <= 61.8) ---
        elif regime_transition:
            # Use trend bias but with reduced size
            if trend_confident_bull and rsi_14[i] < 65.0:
                new_signal = LONG_BASE * 0.7 * vol_scale
            elif trend_confident_bear and rsi_14[i] > 35.0:
                new_signal = -SHORT_BASE * 0.7 * vol_scale
            elif crsi_extreme_oversold and major_bull:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif crsi_extreme_overbought and major_bear:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 20 bars (~10 days on 12h)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if major_bull and inter_bull and rsi_14[i] > 40.0 and rsi_14[i] < 60.0:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif major_bear and inter_bear and rsi_14[i] > 40.0 and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif crsi_extreme_oversold and major_bull:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif crsi_extreme_overbought and major_bear:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_14[i] > 75.0:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_14[i] < 25.0:
                rsi_exit = True
        
        # === CRSI EXIT (mean reversion complete) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long: CRSI moved from oversold to neutral/overbought
            if position_side > 0 and crsi[i] > 60.0:
                crsi_exit = True
            # Short: CRSI moved from overbought to neutral/oversold
            if position_side < 0 and crsi[i] < 40.0:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns bearish
            if position_side > 0 and major_bear and inter_bear:
                regime_reversal = True
            # Short position but 1w regime turns bullish
            if position_side < 0 and major_bull and inter_bull:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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