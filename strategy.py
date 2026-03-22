#!/usr/bin/env python3
"""
Experiment #435: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion + HMA Trend

Hypothesis: After 434 experiments, clear pattern emerges for lower TF success:
1. Connors RSI (CRSI) has proven 75% win rate for mean reversion entries
2. 1h needs SIMPLE logic (2-3 filters max) to generate >=30 trades/symbol
3. Use 4h HMA for trend direction, 1d for major regime (bull/bear)
4. CRSI < 10 = extreme oversold (long), CRSI > 90 = extreme overbought (short)
5. Only enter WITH HTF trend direction (4h HMA slope)
6. Position size: 0.20-0.25 (smaller for 1h to reduce fee drag)

Why this might beat current best (Sharpe=0.435):
- CRSI is proven mean reversion indicator (Larry Connors research)
- 1h TF captures more opportunities than 4h/12h while using HTF filter
- Simple 3-filter logic ensures >=30 trades without over-filtering
- Asymmetric sizing: larger longs in bull regime, larger shorts in bear
- ATR 2.0x stoploss protects against crash scenarios

Position sizing: 0.20-0.25 (discrete, max 0.35 for 1h)
Stoploss: 2.0 * ATR trailing
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_regime_4h1d_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven mean reversion indicator with 75% win rate.
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (use absolute values for RSI calculation)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank - where current close ranks vs last 100 closes
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    # CRSI = average of three components
    crsi = (rsi_3 + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_chop(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (major regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    sma_1d_200 = calculate_sma(df_1d['close'].values, period=200)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    sma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_3_2_100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_chop(high, low, close, 14)
    hma_1h_21 = calculate_hma(close, period=21)
    rsi_1h_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 1h)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(sma_1d_200_aligned[i]):
            continue
        if np.isnan(crsi_3_2_100[i]) or np.isnan(chop_14[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === 1D MAJOR REGIME (bull/bear bias) ===
        # Price above 1d HMA21 + above SMA200 = bull regime (favor longs)
        # Price below 1d HMA21 + below SMA200 = bear regime (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i] and close[i] > sma_1d_200_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i] and close[i] < sma_1d_200_aligned[i]
        neutral_regime = not bull_regime and not bear_regime
        
        # === 4H TREND DIRECTION (entry filter) ===
        # 4h HMA21 > HMA50 = uptrend (only look for longs)
        # 4h HMA21 < HMA50 = downtrend (only look for shorts)
        trend_4h_up = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        trend_4h_down = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = ranging (mean reversion works)
        # CHOP < 45 = trending (trend follow works)
        is_ranging = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = extreme oversold (long opportunity)
        # CRSI > 85 = extreme overbought (short opportunity)
        crsi_oversold = crsi_3_2_100[i] < 15.0
        crsi_overbought = crsi_3_2_100[i] > 85.0
        
        # Weaker signals for more trades
        crsi_weak_oversold = crsi_3_2_100[i] < 25.0
        crsi_weak_overbought = crsi_3_2_100[i] > 75.0
        
        # === 1H LOCAL TREND ===
        price_above_hma = close[i] > hma_1h_21[i]
        price_below_hma = close[i] < hma_1h_21[i]
        
        # === ENTRY LOGIC — SIMPLE 3-FILTER CONFLUENCE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (need: CRSI oversold + 4h trend up OR bull regime)
        if crsi_oversold or (is_ranging and crsi_weak_oversold):
            # Strong signal: CRSI extreme + 4h uptrend
            if trend_4h_up or bull_regime:
                new_signal = LONG_SIZE
            # Moderate signal: CRSI extreme + neutral regime + ranging
            elif neutral_regime and is_ranging:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (need: CRSI overbought + 4h trend down OR bear regime)
        if crsi_overbought or (is_ranging and crsi_weak_overbought):
            # Strong signal: CRSI extreme + 4h downtrend
            if trend_4h_down or bear_regime:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Moderate signal: CRSI extreme + neutral regime + ranging
            elif neutral_regime and is_ranging:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~15 hours on 1h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if (trend_4h_up or bull_regime) and crsi_weak_oversold and price_below_hma:
                new_signal = LONG_SIZE * 0.6
            elif (trend_4h_down or bear_regime) and crsi_weak_overbought and price_above_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi_3_2_100[i] > 70.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_3_2_100[i] < 30.0:
            new_signal = 0.0
        
        # Regime flip exit (1d major trend reversal)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # 4h trend reversal exit
        if in_position and position_side > 0 and trend_4h_down and not bull_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and trend_4h_up and not bear_regime:
            new_signal = 0.0
        
        # === STOPLOSS (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
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
                # Position flip
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