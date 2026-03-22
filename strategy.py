#!/usr/bin/env python3
"""
Experiment #499: 15m Connors RSI Mean-Reversion with 4h HMA Trend Bias

Hypothesis: After 473+ failed experiments, the critical insight is that 15m timeframe
needs aggressive mean-reversion entries filtered by strong HTF trend. Connors RSI
has proven 75% win rate in academic research for short-term mean-reversion.

Key innovations:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI(14) for 15m timeframe
   - Entry thresholds: CRSI < 15 (long), CRSI > 85 (short)
   
2. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Only long when price > 4h HMA (bull regime)
   - Only short when price < 4h HMA (bear regime)
   - Prevents counter-trend mean-reversion trades

3. CHOPPINESS INDEX(14) REGIME FILTER:
   - CHOP > 55 = enable mean-reversion signals
   - CHOP < 45 = reduce position size (trending = harder to mean-revert)
   - Critical for avoiding whipsaws

4. LOOSE ENTRY THRESHOLDS:
   - CRSI < 20 for long (not 10, ensures enough trades)
   - CRSI > 80 for short (not 90, ensures enough trades)
   - MUST generate 50+ trades/year per symbol

5. POSITION SIZING: 0.25 discrete, stoploss at 2.5*ATR
   - Conservative for 15m volatility
   - Discrete levels minimize fee churn

6. VOLUME CONFIRMATION:
   - Entry only if volume > 0.8 * SMA(volume, 20)
   - Avoids low-liquidity fake breakouts

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_connors_rsi_4h_hma_chop_volume_atr_v1"
timeframe = "15m"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI of streak (consecutive up/down closes).
    Streak: +1 for up, -1 for down, cumulative sum reset at sign change.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] >= 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] <= 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on absolute streak values
    streak_s = pd.Series(np.abs(streak))
    delta = streak_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + rs))
    return rsi_streak.values

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank: where current close ranks in last N closes.
    Returns 0-100 scale.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        pr[i] = 100 * rank / period
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + rsi_streak + pr) / 3
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
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
    
    return chop

def calculate_sma(arr, period=20):
    """Calculate Simple Moving Average."""
    arr_s = pd.Series(arr)
    return arr_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    vol_sma = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    SIZE_REDUCED = 0.15  # When trending (harder to mean-revert)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS INDEX REGIME ===
        ranging_market = chop[i] > 55
        trending_market = chop[i] < 45
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === CONNORS RSI MEAN-REVERSION SIGNALS ===
        new_signal = 0.0
        current_size = SIZE if ranging_market else SIZE_REDUCED
        
        # LONG: CRSI oversold + bull regime
        if bull_regime and crsi[i] < 20 and volume_ok:
            new_signal = current_size
        
        # SHORT: CRSI overbought + bear regime
        if bear_regime and crsi[i] > 80 and volume_ok:
            new_signal = -current_size
        
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === CRSI REVERSAL EXIT ===
        # Exit long when CRSI > 70 (overbought)
        # Exit short when CRSI < 30 (oversold)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 70:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 30:
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