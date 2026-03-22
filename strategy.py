#!/usr/bin/env python3
"""
Experiment #174: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Choppiness Regime + Connors RSI

Hypothesis: Previous 4h strategies failed because they used static indicators (EMA/HMA) that
whipsaw in range markets. KAMA (Kaufman Adaptive Moving Average) adapts to volatility —
fast in trends, slow in chop. Combined with Choppiness Index regime detection, this should:
1. Trend-follow when CHOP < 40 (use KAMA direction)
2. Mean-revert when CHOP > 60 (use Connors RSI extremes)
3. Use 12h/1d HTF HMA for major trend bias (avoid counter-trend trades)
4. Looser entry thresholds to ensure 30+ trades/year (learned from 0-trade failures)

Why this should work:
- KAMA has proven edge in crypto (adapts to BTC's vol regimes)
- Choppiness filter prevents trend strategies in range markets (#1 failure mode)
- Connors RSI with thresholds 20/80 (not 10/90) ensures trades trigger
- 4h timeframe = 20-50 trades/year target (low fee drag, proven TF)
- Multiple entry paths (trend + mean revert) ensures trade frequency

Timeframe: 4h (REQUIRED)
HTF: 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_connors_12h1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise — fast in trends, slow in chop.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast - slow) + slow]^2
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    net_change = np.abs(close_s.diff(er_period))
    total_change = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / total_change.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_values = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = rsi_3.fillna(50).values
    
    # Component 2: RSI of Streak
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Streak RSI using 2-period RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / streak_avg_loss.replace(0, np.nan)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50).values
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_30 = calculate_kama(close, er_period=10, fast_period=5, slow_period=40)
    
    # KAMA trend direction
    kama_trend = kama_10 - kama_30
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(kama_trend[i]):
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.2
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.2
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        
        # Strong bias when both agree
        strong_bull = trend_12h_bullish and trend_1d_bullish
        strong_bear = trend_12h_bearish and trend_1d_bearish
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === KAMA TREND ===
        kama_bullish = kama_trend[i] > 0
        kama_bearish = kama_trend[i] < 0
        
        # === CONNORS RSI (looser thresholds for more trades) ===
        crsi_oversold = crsi[i] < 30
        crsi_overbought = crsi[i] > 70
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths (looser for more trades)
        long_score = 0
        
        # Path 1: Trend market + KAMA bullish + HTF bullish bias
        if is_trend_market and kama_bullish and (trend_12h_bullish or trend_1d_bullish):
            long_score += 3
        
        # Path 2: Range market + CRSI oversold (mean revert)
        if is_range_market and crsi_oversold:
            long_score += 3
        
        # Path 3: CRSI extreme low (capitulation)
        if crsi_extreme_low:
            long_score += 2
        
        # Path 4: KAMA bullish + CRSI pullback
        if kama_bullish and crsi[i] < 40:
            long_score += 2
        
        # Path 5: Strong HTF bull + any CRSI oversold
        if strong_bull and crsi[i] < 35:
            long_score += 2
        
        # Path 6: Simple KAMA cross up
        if kama_bullish and kama_trend[i-1] <= 0:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 40:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Trend market + KAMA bearish + HTF bearish bias
        if is_trend_market and kama_bearish and (trend_12h_bearish or trend_1d_bearish):
            short_score += 3
        
        # Path 2: Range market + CRSI overbought (mean revert)
        if is_range_market and crsi_overbought:
            short_score += 3
        
        # Path 3: CRSI extreme high
        if crsi_extreme_high:
            short_score += 2
        
        # Path 4: KAMA bearish + CRSI rally
        if kama_bearish and crsi[i] > 60:
            short_score += 2
        
        # Path 5: Strong HTF bear + any CRSI overbought
        if strong_bear and crsi[i] > 65:
            short_score += 2
        
        # Path 6: Simple KAMA cross down
        if kama_bearish and kama_trend[i-1] >= 0:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if strong_bull and crsi[i] < 40:
                new_signal = current_size * 0.4
            elif strong_bear and crsi[i] > 60:
                new_signal = -current_size * 0.4
            elif crsi[i] < 25:
                new_signal = current_size * 0.3
            elif crsi[i] > 75:
                new_signal = -current_size * 0.3
            elif kama_bullish:
                new_signal = current_size * 0.25
            elif kama_bearish:
                new_signal = -current_size * 0.25
        
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
            if position_side > 0 and is_trend_market and kama_bearish:
                regime_reversal = True
            if position_side < 0 and is_trend_market and kama_bullish:
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