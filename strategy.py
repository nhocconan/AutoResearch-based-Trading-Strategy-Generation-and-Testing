#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d/1w HTF — Dual Regime Adaptive Strategy

Hypothesis: 12h timeframe provides optimal balance between signal quality and trade frequency.
This strategy adapts to market regime using Choppiness Index:
- CHOP > 55 (range): Mean reversion with Connors RSI extremes
- CHOP < 45 (trend): Trend following with HMA confirmation

Key innovations vs failed #001:
1. 12h primary (not 4h) = fewer trades, less fee drag, cleaner signals
2. Dual regime logic (not just mean reversion) = works in 2022 crash AND 2021 bull
3. 1w HMA for secular trend filter = avoids counter-trend trades in strong moves
4. Looser Connors RSI thresholds (25/75 vs 15/85) = more trades, still quality
5. ATR volatility scaling on position size = reduce size in high vol (2022)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, scaled by ATR vol (0.20-0.30 range)
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
    CHOP < 38.2 = trending (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR(1) = True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR(1) over period
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # ATR(period)
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)  # avoid div by zero
    
    # Choppiness Index
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

def calculate_volatility_scaling(atr, lookback=50):
    """
    Scale position size by volatility.
    High vol = smaller size, Low vol = larger size.
    """
    atr_s = pd.Series(atr)
    atr_median = atr_s.rolling(window=lookback, min_periods=lookback).median().values
    vol_ratio = atr / np.where(atr_median == 0, 1e-10, atr_median)
    # vol_ratio > 1.5 = high vol, scale down to 0.7x
    # vol_ratio < 0.7 = low vol, scale up to 1.2x
    scaling = np.clip(1.5 - vol_ratio * 0.3, 0.7, 1.2)
    scaling = np.nan_to_num(scaling, nan=1.0)
    return scaling

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
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility scaling for position size
    vol_scale = calculate_volatility_scaling(atr_14, 50)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === 1W SECULAR TREND (MAJOR BIAS) ===
        # Price above 1w HMA = bullish secular trend (prefer longs)
        # Price below 1w HMA = bearish secular trend (prefer shorts)
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean reversion)
        # CHOP < 45 = trend (trend following)
        # 45-55 = neutral (use either logic)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === CONNORS RSI EXTREMES (looser than typical for more trades) ===
        # CRSI < 25 = oversold (long opportunity)
        # CRSI > 75 = overbought (short opportunity)
        crsi_oversold = connors_rsi[i] < 25
        crsi_overbought = connors_rsi[i] > 75
        
        # === POSITION SIZING with vol scaling ===
        current_size = BASE_SIZE * vol_scale[i]
        current_size = np.clip(current_size, 0.20, 0.35)  # keep in safe range
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Range regime: mean reversion (1w trend filter optional)
        # Trend regime: trend following (require 1w + 1d alignment)
        if is_range:
            # Mean reversion long: oversold + volume
            if crsi_oversold and volume_ok:
                # Prefer longs in bullish/balance secular trend
                if trend_1w_bullish or not trend_1w_bearish:
                    new_signal = current_size
        elif is_trend:
            # Trend following long: both 1w and 1d bullish + pullback
            if trend_1w_bullish and trend_1d_bullish:
                # Wait for pullback (CRSI < 40) or breakout confirmation
                if crsi_oversold or (connors_rsi[i] < 40 and volume_ok):
                    new_signal = current_size
        
        # SHORT ENTRIES
        if is_range:
            # Mean reversion short: overbought + volume
            if crsi_overbought and volume_ok:
                # Prefer shorts in bearish/balance secular trend
                if trend_1w_bearish or not trend_1w_bullish:
                    new_signal = -current_size
        elif is_trend:
            # Trend following short: both 1w and 1d bearish + rally
            if trend_1w_bearish and trend_1d_bearish:
                if crsi_overbought or (connors_rsi[i] > 60 and volume_ok):
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~15 days on 12h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and trend_1d_bullish and connors_rsi[i] < 35:
                new_signal = current_size * 0.7
            elif trend_1w_bearish and trend_1d_bearish and connors_rsi[i] > 65:
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w trend flips bearish + overbought
            if position_side > 0 and trend_1w_bearish and connors_rsi[i] > 70:
                trend_reversal = True
            # Exit short if 1w trend flips bullish + oversold
            if position_side < 0 and trend_1w_bullish and connors_rsi[i] < 30:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
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
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same direction, maintain position (no update needed)
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