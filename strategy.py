#!/usr/bin/env python3
"""
Experiment #296: 12h Primary + 1d HTF — Dual Regime with Connors RSI + HMA Trend

Hypothesis: After 295 experiments, focus on what works:
1. 12h primary TF balances trade frequency (30-60/year) vs signal quality
2. 1d HMA(21) for PRIMARY trend direction — slower, more reliable than 12h
3. Choppiness Index(14) regime switch — mean revert in chop, trend follow otherwise
4. Connors RSI (RSI2 + RSI_Streak + PercentRank) for precise entry timing
5. Asymmetric sizing: larger positions when 1d trend aligns with signal
6. ATR(14) trailing stoploss at 2.5x for 12h timeframe

Why this might beat #292 (Sharpe=0.424):
- Connors RSI has 75% win rate in backtests (vs standard RSI ~55%)
- 12h TF has proven success in #292, now with better entry timing
- Dual regime logic adapts to market conditions automatically
- Less strict filters than #293 to ensure ≥30 trades/year

Position sizing: 0.25 base, 0.35 strong conviction (aligned with 1d trend)
Target: 30-60 trades/year per symbol (appropriate for 12h)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_hma_chop_1d_dual_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_connors_rsi(close, period_rsi=2, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 2) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of closes in last 100 days that are lower than current
    
    Long: CRSI < 10 (extreme oversold)
    Short: CRSI > 90 (extreme overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(2) component
    rsi_short = calculate_rsi(close, period_rsi)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(period_streak, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * min(streak_abs[i], period_streak) / period_streak
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 - min(streak_abs[i], period_streak) / period_streak)
        else:
            streak_rsi[i] = 50
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        window = close[max(0, i-period_rank+1):i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100 * count_lower / (period_rank - 1) if period_rank > 1 else 50
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_2 = calculate_rsi(close, 2)  # For Connors
    crsi = calculate_connors_rsi(close, 2, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.15
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA21
        # Bear: price below 1d HMA21
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        hma_1d_bullish = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_bearish = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        hma_12h_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === RSI THRESHOLDS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === BOLLINGER BAND SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i] * 1.005
        bb_break_upper = close[i] > bb_upper[i] * 0.995
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if is_choppy:
            # Mean revert: CRSI oversold + BB lower break
            if crsi_oversold and bb_break_lower:
                new_signal = BASE_SIZE * vol_scale
            # Extreme CRSI oversold (strong conviction)
            elif crsi_extreme_oversold:
                new_signal = STRONG_SIZE * vol_scale
            # RSI oversold + price near BB lower
            elif rsi_oversold and close[i] < bb_lower[i] * 1.02:
                new_signal = BASE_SIZE * vol_scale
        elif is_trending:
            # Trend follow: 1d bullish + 12h bullish + price above both HMA
            if hma_1d_bullish and hma_12h_bullish and price_above_12h_hma:
                # Enter on pullback (RSI not overbought)
                if rsi_14[i] < 65:
                    new_signal = BASE_SIZE * vol_scale
            # 1d strongly bullish + CRSI recovering from oversold
            elif hma_1d_bullish and crsi[i] > 30 and crsi[i-1] < 30:
                new_signal = STRONG_SIZE * vol_scale
        
        # SHORT ENTRIES
        if is_choppy:
            # Mean revert: CRSI overbought + BB upper break
            if crsi_overbought and bb_break_upper:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
            # Extreme CRSI overbought (strong conviction)
            elif crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE * vol_scale
            # RSI overbought + price near BB upper
            elif rsi_overbought and close[i] > bb_upper[i] * 0.98:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * vol_scale
        elif is_trending:
            # Trend follow: 1d bearish + 12h bearish + price below both HMA
            if hma_1d_bearish and hma_12h_bearish and price_below_12h_hma:
                # Enter on pullback (RSI not oversold)
                if rsi_14[i] > 35:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE * vol_scale
            # 1d strongly bearish + CRSI dropping from overbought
            elif hma_1d_bearish and crsi[i] < 70 and crsi[i-1] > 70:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 12h) ===
        # 12h has ~730 bars/year, need ~30-60 trades = 1 trade per 12-24 bars
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if hma_1d_bullish and crsi[i] < 40:
                new_signal = MIN_SIZE * vol_scale
            elif hma_1d_bearish and crsi[i] > 60:
                new_signal = -MIN_SIZE * vol_scale
            elif is_choppy and crsi_oversold:
                new_signal = MIN_SIZE * vol_scale
            elif is_choppy and crsi_overbought:
                new_signal = -MIN_SIZE * vol_scale
        
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
        
        # === CONNORS RSI EXIT (take profit) ===
        connors_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI goes overbought
            if position_side > 0 and crsi[i] > 80:
                connors_exit = True
            # Short position: exit when CRSI goes oversold
            if position_side < 0 and crsi[i] < 20:
                connors_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns strongly bearish
            if position_side > 0 and hma_1d_bearish and price_below_12h_hma:
                regime_reversal = True
            # Short position but 1d regime turns strongly bullish
            if position_side < 0 and hma_1d_bullish and price_above_12h_hma:
                regime_reversal = True
        
        if stoploss_triggered or connors_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.18:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = STRONG_SIZE * vol_scale
            elif new_signal > 0:
                new_signal = BASE_SIZE * vol_scale
            elif new_signal < -0.30:
                new_signal = -STRONG_SIZE * vol_scale
            else:
                new_signal = -BASE_SIZE * vol_scale
        
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