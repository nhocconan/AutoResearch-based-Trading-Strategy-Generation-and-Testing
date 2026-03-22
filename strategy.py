#!/usr/bin/env python3
"""
Experiment #022: 12h Primary + 1d/1w HTF — Dual Regime Adaptive Strategy

Hypothesis: 12h timeframe with dual regime logic adapts to market conditions.
In choppy/range markets (CHOP>55): mean reversion via Connors RSI extremes.
In trending markets (CHOP<45): trend following via HMA pullback entries.

Key innovations:
1. 1w HMA(21) for SECULAR trend bias (only trade WITH 1w trend)
2. 1d HMA(21) for INTERMEDIATE trend confirmation
3. Choppiness Index(14) regime switch at 12h level
4. Connors RSI(3,2,100) for mean reversion entries in range regime
5. HMA(16/48) crossover + RSI(14) pullback for trend entries
6. ATR(14) trailing stoploss at 2.5x
7. Volume confirmation filter (>0.7x 20-bar avg)

Why this should work on 12h:
- 12h candles filter out intraday noise (20-50 trades/year target)
- Regime detection prevents trend-following in ranges (major failure mode)
- 1w bias prevents counter-secular-trend trades
- Connors RSI has 75% win rate for extremes
- Discrete sizing (0.25/0.30) minimizes fee churn

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_connors_hma_1d1w_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down bars
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50).values
    
    # Component 3: Percent Rank
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= rank_period else 50
    )
    percent_rank = percent_rank.fillna(50).values
    
    # Connors RSI
    connors_rsi = (rsi_3 + streak_rsi + percent_rank) / 3
    return connors_rsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 61.8 = range/choppy (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    chop = 100 * (atr1_sum / atr_period) / hh_ll * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    connors_rsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # HMA for trend following (16/48 crossover)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Bands for mean reversion targets
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === 1W SECULAR TREND BIAS (MAJOR) ===
        # Only trade WITH 1w trend direction
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean reversion preferred)
        # CHOP < 45 = trend (trend following preferred)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === CONNORS RSI EXTREMES (for range regime) ===
        # CRSI < 15 = deeply oversold (long)
        # CRSI > 85 = deeply overbought (short)
        crsi_oversold = connors_rsi[i] < 15
        crsi_overbought = connors_rsi[i] > 85
        
        # === HMA CROSSOVER (for trend regime) ===
        hma_bullish_cross = hma_16[i] > hma_48[i]
        hma_bearish_cross = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK (for trend entries) ===
        # In uptrend: RSI pullback to 40-50
        # In downtrend: RSI rally to 50-60
        rsi_pullback_long = 35 < rsi_14[i] < 55
        rsi_pullback_short = 45 < rsi_14[i] < 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === RANGE REGIME: MEAN REVERSION ===
        if is_range:
            # Long: 1w bullish bias + oversold + near BB lower
            if trend_1w_bullish and crsi_oversold and close[i] < bb_lower[i] * 1.005:
                if volume_ok:
                    new_signal = current_size
            
            # Short: 1w bearish bias + overbought + near BB upper
            if trend_1w_bearish and crsi_overbought and close[i] > bb_upper[i] * 0.995:
                if volume_ok:
                    new_signal = -current_size
        
        # === TREND REGIME: TREND FOLLOWING ===
        if is_trend:
            # Long: 1w bullish + 1d bullish + HMA bullish + RSI pullback
            if trend_1w_bullish and trend_1d_bullish and hma_bullish_cross:
                if rsi_pullback_long and volume_ok:
                    new_signal = current_size
            
            # Short: 1w bearish + 1d bearish + HMA bearish + RSI pullback
            if trend_1w_bearish and trend_1d_bearish and hma_bearish_cross:
                if rsi_pullback_short and volume_ok:
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~150 days on 12h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            # Weaker conditions: just need 1w trend + extreme CRSI
            if trend_1w_bullish and connors_rsi[i] < 25:
                new_signal = current_size * 0.7
            elif trend_1w_bearish and connors_rsi[i] > 75:
                new_signal = -current_size * 0.7
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trend and not trend_1w_bullish:
                regime_exit = True
            if position_side < 0 and is_trend and not trend_1w_bearish:
                regime_exit = True
        
        # === TAKE PROFIT (reduce to half at 2R) ===
        take_profit = False
        if in_position and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr_14[i] and signals[i-1] == current_size:
                    take_profit = True
                    new_signal = current_size * 0.5
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr_14[i] and signals[i-1] == -current_size:
                    take_profit = True
                    new_signal = -current_size * 0.5
        
        # Apply stoploss or regime exit (overrides take profit)
        if stoploss_triggered or regime_exit:
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
            elif abs(new_signal) < abs(signals[i-1]) and signals[i-1] != 0:
                # Partial exit (take profit) - keep tracking
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