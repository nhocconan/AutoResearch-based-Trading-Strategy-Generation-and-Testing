#!/usr/bin/env python3
"""
Experiment #447: 1h Connors RSI + Choppiness Regime + 4h HMA Trend

Hypothesis: After 431 failed experiments, 1h strategies fail because they either
chase trends (whipsaw in 2022) or mean-revert blindly (crushed in strong trends).
This strategy uses proven quantitative edges:

1. CONNORS RSI (CRSI) - Larry Connors' proven 75% win rate mean reversion:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 20 (oversold, loosened from 10 for more trades)
   - Short when CRSI > 80 (overbought, loosened from 90 for more trades)
   - Catches short-term extremes on 1h timeframe

2. CHOPPINESS INDEX (CHOP) Regime Filter - Ehlers/QuantConnect research:
   - CHOP(14) > 55 = ranging market (enable mean reversion entries)
   - CHOP(14) < 45 = trending market (only enter on extreme pullbacks)
   - Prevents mean reversion disasters during strong trends

3. 4h HMA(21) Trend Bias (via mtf_data helper):
   - Long bias when price > 4h HMA
   - Short bias when price < 4h HMA
   - HMA smoother than EMA, critical for HTF trend detection

4. ATR(14) Trailing Stop at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

5. Position Sizing: 0.28 discrete (conservative for 1h volatility)

Why this should work on 1h:
- CRSI catches short-term extremes (more trades than daily strategies)
- CHOP filter prevents disaster during strong trends
- 4h HMA provides directional bias without being too slow
- Looser CRSI thresholds (20/80 vs 10/90) ensure sufficient trade frequency
- Should generate 30-60 trades/year per symbol (well above 10 minimum)

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h_hma_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_lookback=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with ~75% win rate.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - fast RSI component
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    up_streak = np.zeros(n)
    down_streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            up_streak[i] = up_streak[i-1] + 1
            down_streak[i] = 0
        elif close[i] < close[i-1]:
            down_streak[i] = down_streak[i-1] + 1
            up_streak[i] = 0
        else:
            up_streak[i] = up_streak[i-1]
            down_streak[i] = down_streak[i-1]
    
    # RSI of streak values (using 2-period)
    streak_values = up_streak - down_streak
    streak_values_s = pd.Series(streak_values)
    delta = streak_values_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    streak_rsi = 100 - (100 / (1 + rs))
    streak_rsi = streak_rsi.values
    
    # Percent Rank (100) - where current close ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_lookback, n):
        window = close[i-rank_lookback:i]
        current = close[i]
        count_lower = np.sum(window < current)
        percent_rank[i] = 100 * count_lower / rank_lookback
    
    # CRSI = average of three components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    Using 55/45 thresholds for more sensitivity on 1h.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR first
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        atr_sum = atr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        ranging_market = chop[i] > 55  # Looser threshold for more trades
        trending_market = chop[i] < 45  # Clear trend detection
        
        # === CONNORS RSI SIGNALS ===
        # Looser thresholds (20/80 vs 10/90) to ensure sufficient trades
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        
        # Extreme signals for trending markets
        crsi_extreme_oversold = crsi[i] < 15
        crsi_extreme_overbought = crsi[i] > 85
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Mean reversion: primary signal in ranging market, aligned with 4h trend
        if ranging_market:
            if crsi_oversold and bull_trend_4h:
                new_signal = SIZE
            elif crsi_overbought and bear_trend_4h:
                new_signal = -SIZE
        
        # In trending market: only enter on extreme pullbacks (stricter)
        if trending_market and new_signal == 0.0:
            if crsi_extreme_oversold and bull_trend_4h:
                new_signal = SIZE
            elif crsi_extreme_overbought and bear_trend_4h:
                new_signal = -SIZE
        
        # Neutral choppiness (45-55): allow both types of entries
        if not ranging_market and not trending_market and new_signal == 0.0:
            if crsi_oversold and bull_trend_4h:
                new_signal = SIZE
            elif crsi_overbought and bear_trend_4h:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals