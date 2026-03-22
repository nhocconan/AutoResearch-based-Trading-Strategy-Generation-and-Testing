#!/usr/bin/env python3
"""
Experiment #031: 15m Connors RSI Mean Reversion + 4h HMA Trend Filter + Choppiness Regime

Hypothesis: After analyzing 26 failed strategies, the pattern is clear:
1. 15m timeframe is TOO NOISY for pure trend following (all 15m attempts failed: Sharpe -2 to -3)
2. Mean reversion works BETTER on lower timeframes than trend following
3. Connors RSI (CRSI) has 75% win rate in academic studies for short-term mean reversion
4. MUST use 4h HMA as strong trend filter - only trade in HTF direction
5. Choppiness Index (CHOP) detects regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend
6. Volume spike confirmation filters fake mean reversion signals
7. Conservative sizing (0.20-0.30) + 2.5*ATR stop to survive 2022-style crashes

Strategy components:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive than standard RSI, catches short-term extremes
   - Long entry: CRSI < 15 (oversold), Short entry: CRSI > 85 (overbought)

2. 4h HMA(21) trend filter: ONLY long if price > 4h HMA, ONLY short if price < 4h HMA
   - Prevents counter-trend mean reversion (major failure mode)

3. Choppiness Index(14): Regime detection
   - CHOP > 55: Range regime (enable mean reversion entries)
   - CHOP < 45: Trend regime (disable mean reversion, stay flat or trend follow)
   - This avoids mean reversion during strong trends where it fails

4. Volume confirmation: Volume > 1.5x 20-bar average
   - Confirms genuine reversal interest, not just noise

5. ATR(14) trailing stop: 2.5*ATR from entry
   - Tighter than 12h strategies (3.0*ATR) due to 15m noise
   - Exit signal when stop hit

6. Position sizing: 0.25 base, 0.30 with all confirmations
   - Discrete levels to minimize fee churn

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Target trades: 40-80/year (optimal per Rule 10 for 15m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_chop_regime_vol_atr_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of today's price change over lookback period
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - short-term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of streaks - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak values
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank of price change over lookback period
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        changes = np.diff(close[i-rank_period:i+1])
        current_change = changes[-1] if len(changes) > 0 else 0
        if len(changes) > 0:
            pct_rank[i] = np.sum(changes[:-1] <= current_change) / (len(changes) - 1) * 100
        else:
            pct_rank[i] = 50
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR for each bar
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    
    # Sum of ATR over period
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = hh - ll
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_volume_spike(volume, lookback=20, threshold=1.5):
    """Detect volume spikes above threshold * average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=lookback, min_periods=lookback).mean()
    vol_spike = volume > (threshold * vol_avg)
    return vol_spike

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
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    vol_spike = calculate_volume_spike(volume, lookback=20, threshold=1.5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_vs_4h = close[i] - hma_4h_aligned[i]
        bull_htf = price_vs_4h > 0
        bear_htf = price_vs_4h < 0
        
        # === CHOPPINESS REGIME FILTER ===
        # Only enable mean reversion in range/choppy markets
        chop_range = chop[i] > 55  # Range regime - enable mean reversion
        chop_trend = chop[i] < 45  # Trend regime - disable mean reversion
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Strong oversold
        crsi_overbought = crsi[i] > 85  # Strong overbought
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRY: HTF bull + range regime + CRSI oversold + volume confirmation
        if bull_htf and chop_range and crsi_oversold:
            signal_strength += 2  # HTF trend + CRSI extreme (core signals)
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            # Assign size based on confirmation count
            if signal_strength >= 3:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # SHORT ENTRY: HTF bear + range regime + CRSI overbought + volume confirmation
        elif bear_htf and chop_range and crsi_overbought:
            signal_strength += 2  # HTF trend + CRSI extreme (core signals)
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            # Assign size based on confirmation count
            if signal_strength >= 3:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        # === REGIME CHANGE EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit if regime changes to strong trend (mean reversion fails in trends)
            if chop_trend:
                regime_exit = True
        
        # Apply stoploss or exit conditions
        if stoploss_triggered or crsi_exit or regime_exit:
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