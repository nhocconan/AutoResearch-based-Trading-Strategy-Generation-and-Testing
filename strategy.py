#!/usr/bin/env python3
"""
Experiment #363: 1h Connors RSI Mean Reversion with 4h HMA Trend + Choppiness Regime Filter

Hypothesis: After 362 experiments, the clearest pattern is that pure trend-following fails on
BTC/ETH in bear/range markets (2022 crash, 2025 bear). Mean reversion with regime filtering
shows promise from research literature.

Key components:
1. CONNORS RSI (CRSI): Composite of RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - More sensitive than standard RSI(14) for catching extremes
   - Long when CRSI < 15 (oversold), Short when CRSI > 85 (overbought)
   - Research shows 70-75% win rate on crypto mean reversion

2. CHOPPINESS INDEX (CHOP) REGIME FILTER:
   - CHOP > 61.8 = ranging market (enable mean reversion entries)
   - CHOP < 38.2 = trending market (disable mean reversion, avoid whipsaw)
   - Critical filter: mean reversion ONLY works in choppy markets

3. 4h HMA TREND BIAS (via mtf_data helper):
   - Long CRSI signals only if price > 4h HMA(21) (bullish bias)
   - Short CRSI signals only if price < 4h HMA(21) (bearish bias)
   - Filters counter-trend mean reversion (dangerous in strong trends)

4. ATR TRAILING STOP (2.5x):
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from mean reversion failures (trend continuation)

5. POSITION SIZING: 0.30 discrete (conservative for 1h volatility)
   - Max 30% capital per position
   - Discrete levels minimize fee churn

Why 1h should work:
- Faster than 4h/12h strategies (more trade opportunities)
- CRSI catches intraday extremes that daily strategies miss
- CHOP filter avoids trend whipsaw (major failure mode of past strategies)
- 4h HMA provides stable bias without excessive lag
- Should generate 30-60 trades/year per symbol (enough for stats, not too many for fees)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_chop_regime_atr_v1"
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
    """Calculate RSI using standard Wilder's method."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(len(close))
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss <= 1e-10] = 100.0
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of CRSI.
    Streak = consecutive up/down days, normalized to 0-100 scale.
    """
    n = len(close)
    streak_rsi = np.zeros(n)
    
    for i in range(period, n):
        # Count consecutive up/down streaks
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(0, i - period * 3), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        # Net streak (positive = up, negative = down)
        net_streak = up_streak - down_streak
        
        # Normalize to 0-100 scale (period=2 means max streak ~6)
        max_streak = period * 3
        streak_rsi[i] = 50 + (net_streak / max_streak) * 50
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of CRSI.
    PercentRank = percentage of closes in lookback period that are below current close.
    """
    n = len(close)
    pr = np.zeros(n)
    
    for i in range(period, n):
        lookback = close[i-period+1:i+1]
        count_below = np.sum(lookback < close[i])
        pr[i] = (count_below / period) * 100
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + rsi_streak + percent_rank) / 3
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
            chop[i] = np.clip(chop[i], 0, 100)
    
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
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Need 150 bars for CRSI percent_rank(100) + warmup
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS INDEX REGIME FILTER ===
        # Only allow mean reversion in choppy/ranging markets
        choppy_market = chop[i] > 55.0  # Slightly lowered from 61.8 to get more trades
        trending_market = chop[i] < 40.0
        
        # === CRSI EXTREME SIGNALS ===
        # Long: CRSI < 15 (oversold)
        crsi_oversold = crsi[i] < 15.0
        
        # Short: CRSI > 85 (overbought)
        crsi_overbought = crsi[i] > 85.0
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + choppy market + 4h bullish bias
        if crsi_oversold and choppy_market and bull_trend_4h:
            new_signal = SIZE
        
        # SHORT ENTRY: CRSI overbought + choppy market + 4h bearish bias
        elif crsi_overbought and choppy_market and bear_trend_4h:
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
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long when CRSI returns to neutral (> 50)
        if in_position and position_side > 0 and crsi[i] > 50.0:
            new_signal = 0.0
        
        # Exit short when CRSI returns to neutral (< 50)
        if in_position and position_side < 0 and crsi[i] < 50.0:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === MARKET REGIME CHANGE EXIT ===
        # Exit if market becomes strongly trending (mean reversion dangerous)
        if in_position and trending_market:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals