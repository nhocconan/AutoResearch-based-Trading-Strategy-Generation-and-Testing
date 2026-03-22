#!/usr/bin/env python3
"""
Experiment #133: 1d Primary + 1w HTF — Dual Regime Mean Reversion + Trend

Hypothesis: Daily timeframe with weekly trend bias should capture major moves
while avoiding whipsaw. Previous 12h/4h strategies failed due to too many
conditions filtering out all trades. This strategy simplifies entry logic:

1. WEEKLY HMA(21) SLOPE: Major trend bias (bull/bear regime)
2. DAILY CONNORS RSI: Entry timing (oversold<20 long, overbought>80 short)
3. CHOPPINESS INDEX: Regime switch (chop>55 mean-revert, chop<45 trend-follow)
4. ATR TRAILING STOP: 2.5*ATR exit on adverse moves
5. ASYMMETRIC SIZING: Larger positions with trend, smaller against

Why this should work:
- 1d timeframe = 20-40 trades/year (low fee drag, matches target)
- 1w HTF prevents fighting major trends
- Connors RSI has proven 75% win rate for extremes
- Simpler logic = more trades (avoiding 0-trade failure)
- Works in both bull (2021) and bear (2022, 2025) markets

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Target trades: 20-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_chop_hma_1w_v1"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 15)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 15)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # RSI(14) for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    # 200-day SMA for major trend
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    TREND_SIZE = 0.32
    MEAN_REV_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(250, n):  # Start after 250 bars for SMA200
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55
        is_trending = chop_14[i] < 45
        
        # === CONNORS RSI EXTREMES (looser thresholds for more trades) ===
        crsi_oversold = crsi[i] < 22
        crsi_overbought = crsi[i] > 78
        crsi_extreme_low = crsi[i] < 15
        crsi_extreme_high = crsi[i] > 85
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_trending and weekly_bullish:
            current_size = TREND_SIZE
        elif is_choppy:
            current_size = MEAN_REV_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Simplified for more trades
        long_signal = False
        
        # Path 1: CRSI oversold + weekly bullish (primary)
        if crsi_oversold and weekly_bullish:
            long_signal = True
        
        # Path 2: CRSI extreme + price above SMA200 (pullback in bull)
        if crsi_extreme_low and price_above_sma200:
            long_signal = True
        
        # Path 3: Choppy market + CRSI oversold (mean revert)
        if is_choppy and crsi_oversold and rsi_oversold:
            long_signal = True
        
        # Path 4: Weekly bullish + RSI oversold (trend pullback)
        if weekly_bullish and rsi_oversold and bars_since_last_trade > 30:
            long_signal = True
        
        if long_signal:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_signal = False
        
        # Path 1: CRSI overbought + weekly bearish (primary)
        if crsi_overbought and weekly_bearish:
            short_signal = True
        
        # Path 2: CRSI extreme + price below SMA200 (rally in bear)
        if crsi_extreme_high and price_below_sma200:
            short_signal = True
        
        # Path 3: Choppy market + CRSI overbought (mean revert)
        if is_choppy and crsi_overbought and rsi_overbought:
            short_signal = True
        
        # Path 4: Weekly bearish + RSI overbought (trend retracement)
        if weekly_bearish and rsi_overbought and bars_since_last_trade > 30:
            short_signal = True
        
        if short_signal:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD — Force trades if too quiet ===
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            # Force entry on CRSI extremes
            if crsi[i] < 18 and (weekly_bullish or price_above_sma200):
                new_signal = current_size * 0.5
            elif crsi[i] > 82 and (weekly_bearish or price_below_sma200):
                new_signal = -current_size * 0.5
        
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
            # Exit long if weekly turns bearish strongly
            if position_side > 0 and hma_1w_slope_aligned[i] < -1.0:
                regime_reversal = True
            # Exit short if weekly turns bullish strongly
            if position_side < 0 and hma_1w_slope_aligned[i] > 1.0:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === CRSI REVERSAL EXIT ===
        # Exit long when CRSI goes overbought
        if in_position and position_side > 0 and crsi[i] > 80:
            new_signal = 0.0
        # Exit short when CRSI goes oversold
        if in_position and position_side < 0 and crsi[i] < 20:
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