#!/usr/bin/env python3
"""
Experiment #420: 1d Connors RSI + Weekly HMA Trend + Choppiness Regime Filter

Hypothesis: After 419 failed experiments, the key insight is that 1d timeframe
needs MEAN REVERSION with strong trend bias, not pure trend following.
BTC/ETH 2022-2025 are mostly bear/range markets where trend strategies fail.

This strategy uses:

1. CONNORS RSI (CRSI) - Proven 75% win rate for mean reversion:
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 (extreme oversold)
   - Short when CRSI > 90 (extreme overbought)
   - Much more reliable than standard RSI(14)

2. WEEKLY HMA(21) TREND BIAS (via mtf_data helper):
   - Long ONLY when price > 1w HMA (bullish bias)
   - Short ONLY when price < 1w HMA (bearish bias)
   - HMA smoother than EMA, critical for weekly alignment
   - This prevents counter-trend mean reversion (deadly in crashes)

3. CHOPPINESS INDEX REGIME FILTER:
   - CHOP(14) > 61.8 = ranging market (enable mean reversion)
   - CHOP(14) < 38.2 = trending market (reduce position or skip)
   - Best meta-filter for identifying when mean reversion works

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crashes

5. POSITION SIZING: 0.25 discrete (conservative for 1d volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 1d should work now:
- Connors RSI specifically designed for daily mean reversion
- Weekly HMA provides major trend filter (avoid counter-trend)
- Choppiness Index filters out trending periods where mean reversion fails
- Should generate 20-40 trades/year (enough for stats, not too many for fees)
- Works on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_rsi_weekly_hma_chop_regime_atr_v1"
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

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    This is proven to have 75% win rate for mean reversion on daily data.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period_rsi)
    
    # Component 2: RSI of Streak Length
    # Streak = consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            if delta.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta.iloc[i] < 0:
            if delta.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # RSI of streak (using absolute values for calculation)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: Percentile Rank of today's return vs last 100 days
    returns = close_s.pct_change()
    percent_rank = np.full(n, np.nan)
    for i in range(period_rank, n):
        window = returns.iloc[i-period_rank+1:i+1]
        current_return = returns.iloc[i]
        if not np.isnan(current_return):
            rank = (window < current_return).sum()
            percent_rank[i] = rank / period_rank * 100
    
    # Combine all three components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar (True Range)
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
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8  # Mean reversion works here
        trending_market = chop[i] < 38.2  # Trend following works here
        # neutral = 38.2 <= CHOP <= 61.8 (reduce size or stay flat)
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_low = crsi[i] < 10   # Extreme oversold
        crsi_extreme_high = crsi[i] > 90  # Extreme overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # RANGING REGIME: Mean reversion with weekly trend bias
        if ranging_market:
            # Long: CRSI extreme low + weekly bullish bias
            if crsi_extreme_low and bull_trend_1w:
                new_signal = SIZE
            # Short: CRSI extreme high + weekly bearish bias
            elif crsi_extreme_high and bear_trend_1w:
                new_signal = -SIZE
        
        # TRENDING REGIME: Reduce position size or skip mean reversion
        elif trending_market:
            # Only enter with strong trend confirmation
            if crsi_extreme_low and bull_trend_1w:
                new_signal = SIZE * 0.6  # Reduced size in trending market
            elif crsi_extreme_high and bear_trend_1w:
                new_signal = -SIZE * 0.6
        
        # NEUTRAL REGIME: Stay flat or small positions
        else:
            # Small positions only with strong CRSI signal
            if crsi_extreme_low and bull_trend_1w:
                new_signal = SIZE * 0.5
            elif crsi_extreme_high and bear_trend_1w:
                new_signal = -SIZE * 0.5
        
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
            # Exit long if weekly trend turns bearish
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            # Exit short if weekly trend turns bullish
            if position_side < 0 and bull_trend_1w:
                new_signal = 0.0
        
        # === CRSI REVERSAL EXIT ===
        # Exit when CRSI crosses back through neutral (50)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 50:
                new_signal = 0.0  # Take profit on long
            if position_side < 0 and crsi[i] < 50:
                new_signal = 0.0  # Take profit on short
        
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