#!/usr/bin/env python3
"""
Experiment #023: 1d Dual-Regime Strategy with 1w Confirmation

Hypothesis: After 22 failed experiments, return to proven patterns from research notes.
This implements a DUAL-REGIME strategy that switches logic based on market state:

1. CHOPPINESS INDEX (CHOP) regime detection:
   - CHOP > 61.8 = ranging market → use Connors RSI mean reversion
   - CHOP < 38.2 = trending market → use HMA + Donchian breakout
   
2. CONNORS RSI for mean reversion (proven on ETH with Sharpe +0.923):
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > SMA(200)
   - Short: CRSI > 90 + price < SMA(200)
   
3. HMA + Donchian for trend following (proven on SOL with Sharpe +0.782):
   - Long: HMA(21) > HMA(48) + Donchian(20) breakout + 1w bullish
   - Short: HMA(21) < HMA(48) + Donchian(20) breakdown + 1w bearish

4. 1W HMA(48) for secular trend bias (filters counter-trend trades)

5. ATR(14) trailing stop at 2.5 ATR (protects capital in 2022-style crashes)

Timeframe: 1d (REQUIRED - generates 20-50 trades/year naturally)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing

Why this should work:
- Regime-switching avoids trend-following whipsaws in chop (2022 bottom)
- Connors RSI has 75% win rate in ranges
- 1d timeframe = natural trade frequency control (not too many fees)
- 1w confirmation filters false signals
- Discrete sizing minimizes fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_connors_hma_donchian_1w_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak values
    streak_series = pd.Series(streak)
    streak_delta = streak_series.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / avg_streak_loss
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # Percent Rank component
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] * 100 if close[i-1] > 0 else 0
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current_return = returns[i]
        rank = np.sum(window < current_return) / rank_period * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W indicators
    hma_1w_48 = calculate_hma(df_1w['close'].values, 48)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_48_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_48)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_48 = calculate_hma(close, 48)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(250, n):  # Start after 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_48_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === 1W SECULAR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_48_aligned[i]
        weekly_bearish = close[i] < hma_1w_48_aligned[i]
        
        # === 1D HMA TREND ===
        hma_bullish = hma_1d_21[i] > hma_1d_48[i]
        hma_bearish = hma_1d_21[i] < hma_1d_48[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 90
        
        # === PRICE VS SMA200 ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === POSITION SIZING ===
        long_size = BASE_SIZE
        short_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE (Connors RSI)
            # Long: CRSI < 10 + price > SMA200 + weekly bullish bias preferred
            if crsi_oversold and above_sma200:
                if weekly_bullish or hma_bullish:
                    new_signal = long_size
            
            # Short: CRSI > 90 + price < SMA200 + weekly bearish bias preferred
            if crsi_overbought and below_sma200:
                if weekly_bearish or hma_bearish:
                    new_signal = -short_size
        else:
            # TREND FOLLOWING MODE (HMA + Donchian)
            # Long: HMA bullish + Donchian breakout + weekly bullish
            if hma_bullish and breakout_long and weekly_bullish:
                new_signal = long_size
            
            # Short: HMA bearish + Donchian breakdown + weekly bearish
            if hma_bearish and breakout_short and weekly_bearish:
                new_signal = -short_size
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if 1d HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals