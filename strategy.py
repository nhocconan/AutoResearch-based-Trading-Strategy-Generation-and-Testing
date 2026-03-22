#!/usr/bin/env python3
"""
Experiment #354: 4h Primary + 12h/1d HTF — Choppiness Regime + Connors RSI + Bollinger Mean Reversion

Hypothesis: After 350+ experiments, the clearest pattern is:
1. Regime detection (Choppiness Index) is CRITICAL - ETH Sharpe +0.923 in exp history
2. Connors RSI (CRSI) has 75% win rate for mean reversion entries
3. Simple logic with fewer AND conditions = more trades = better Sharpe
4. 4h timeframe with proper HTF filter avoids both 15m noise and 1d slowness

This strategy combines:
1. Choppiness Index(14) for regime: CHOP>61.8=range(mean revert), CHOP<38.2=trend(breakout)
2. Connors RSI for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Bollinger Bands(20,2.0) for volatility bounds
4. 12h HMA(21) for intermediate trend filter
5. 1d HMA(21) for major trend bias
6. ATR(14) trailing stop 2.5x for risk management
7. Asymmetric sizing: longs 0.25-0.35, shorts 0.15-0.25 (crypto long bias)

Why this might beat current best (Sharpe=0.435):
- Choppiness regime filter adapts to market conditions (range vs trend)
- Connors RSI catches reversals better than standard RSI(14)
- Dual HTF (12h + 1d) provides better trend context than single HTF
- Simpler entry logic than exp #349 = more trades = better statistics
- Bollinger squeeze detection adds volatility filter

Position sizing: 0.25-0.35 longs, 0.15-0.25 shorts
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h (1 trade every 7-12 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_bb_12h1d_regime_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        if span < 1:
            return series
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR(1), n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = period
    
    # Calculate ATR(1) = true range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR(1) over period
    tr_series = pd.Series(tr)
    atr_sum = tr_series.rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Choppiness calculation
    price_range = hh - ll
    chop = np.zeros(len(close))
    
    valid = (price_range > 0) & (atr_sum > 0)
    chop[valid] = 100.0 * np.log10(atr_sum[valid] / price_range[valid]) / np.log10(n)
    
    return chop

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile of current price vs last 100 prices
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak
    # Streak = consecutive up (+1) or down (-1) days
    direction = np.sign(delta.values)
    direction[0] = 0
    
    streak = np.zeros(n)
    for i in range(1, n):
        if direction[i] == direction[i-1] and direction[i] != 0:
            streak[i] = streak[i-1] + direction[i]
        elif direction[i] != 0:
            streak[i] = direction[i]
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: Percent Rank
    percent_rank = pd.Series(close).rolling(window=period_rank, min_periods=period_rank).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100.0, raw=False
    )
    
    # Combine components
    crsi = (rsi_short.values + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    mid = sma
    
    return upper.values, lower.values, mid.values, std.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # ATR ratio for volatility regime
    atr_ratio = np.divide(atr_14, atr_30 + 1e-10, out=np.ones_like(atr_14), where=atr_30 > 0)
    
    # Choppiness Index for regime detection
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Connors RSI for entry timing
    crsi = calculate_connors_rsi(close)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid, bb_std = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # BB Width for squeeze detection
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_width_pct = pd.Series(bb_width).rolling(window=20, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.35
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.25
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy (favor mean reversion)
        # CHOP < 38.2 = trending (favor breakout/trend follow)
        # 38.2 <= CHOP <= 61.8 = transition (reduce size)
        regime_range = chop[i] > 61.8
        regime_trend = chop[i] < 38.2
        regime_transition = not regime_range and not regime_trend
        
        # Volatility scaling
        high_vol = atr_ratio[i] > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === HTF TREND BIAS ===
        # 12h HMA for intermediate trend
        trend_12h_bull = close[i] > hma_12h_21_aligned[i]
        trend_12h_bear = close[i] < hma_12h_21_aligned[i]
        
        # 1d HMA for major trend bias
        trend_1d_bull = close[i] > hma_1d_21_aligned[i]
        trend_1d_bear = close[i] < hma_1d_21_aligned[i]
        
        # Combined trend score (-2 to +2)
        trend_score = 0
        if trend_12h_bull:
            trend_score += 1
        if trend_1d_bull:
            trend_score += 1
        if trend_12h_bear:
            trend_score -= 1
        if trend_1d_bear:
            trend_score -= 1
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0
        crsi_strong_oversold = crsi[i] < 10.0
        crsi_overbought = crsi[i] > 80.0
        crsi_strong_overbought = crsi[i] > 90.0
        crsi_neutral_long = 25.0 < crsi[i] < 55.0
        crsi_neutral_short = 45.0 < crsi[i] < 75.0
        
        # === BOLLINGER BAND SIGNALS ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_lower = close[i] < bb_lower[i] * 1.005
        price_near_upper = close[i] > bb_upper[i] * 0.995
        
        # BB squeeze (low volatility = potential breakout)
        bb_squeeze = bb_width_pct[i] < 0.2
        
        # === VOLATILITY SPIKE REVERSION ===
        vol_spike = atr_ratio[i] > 2.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if trend_score >= 0:  # Neutral to bullish bias
            # Range regime: Mean reversion at BB lower + CRSI oversold
            if regime_range and price_below_bb_lower and crsi_oversold:
                new_signal = LONG_BASE * vol_scale
            
            # Range regime: Strong oversold (CRSI < 10)
            elif regime_range and crsi_strong_oversold:
                new_signal = LONG_STRONG * vol_scale
            
            # Trend regime: CRSI pullback to neutral + price above BB mid
            elif regime_trend and crsi_neutral_long and close[i] > bb_mid[i]:
                new_signal = LONG_BASE * vol_scale
            
            # Vol spike reversion: ATR spike + price below BB lower + CRSI < 25
            elif vol_spike and price_below_bb_lower and crsi[i] < 25.0:
                new_signal = LONG_BASE * 0.8 * vol_scale
            
            # BB squeeze breakout: squeeze + price breaks above BB upper + trend bullish
            elif bb_squeeze and price_above_bb_upper and trend_score >= 1:
                new_signal = LONG_STRONG * vol_scale
        
        # SHORT ENTRIES (reduced size, only with bearish bias)
        if trend_score <= 0:  # Neutral to bearish bias
            # Range regime: Mean reversion at BB upper + CRSI overbought
            if regime_range and price_above_bb_upper and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Range regime: Strong overbought (CRSI > 90)
            elif regime_range and crsi_strong_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # Trend regime: CRSI pullback to neutral + price below BB mid
            elif regime_trend and crsi_neutral_short and close[i] < bb_mid[i]:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Vol spike reversion: ATR spike + price above BB upper + CRSI > 75
            elif vol_spike and price_above_bb_upper and crsi[i] > 75.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === TRANSITION REGIME (reduce size, wait for clarity) ===
        if regime_transition and new_signal != 0.0:
            new_signal = new_signal * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 20 bars (~3.3 days on 4h)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if trend_score >= 1 and crsi[i] < 40.0:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif trend_score <= -1 and crsi[i] > 60.0:
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but trend turns strongly bearish
            if position_side > 0 and trend_score <= -1:
                trend_reversal = True
            # Short position but trend turns strongly bullish
            if position_side < 0 and trend_score >= 1:
                trend_reversal = True
        
        if stoploss_triggered or crsi_exit or trend_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.20:
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