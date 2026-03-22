#!/usr/bin/env python3
"""
Experiment #382: 4h Connors RSI + 1d HMA Trend + ADX Regime Filter

Hypothesis: After analyzing 381 failed experiments, the key insight is that 
CONNORS RSI (CRSI) has proven 75% win rate in academic literature for mean-reversion
entries, but needs HTF trend confirmation to avoid counter-trend trades in strong trends.

STRATEGY COMPONENTS:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - CRSI < 10 = extreme oversold (long opportunity)
   - CRSI > 90 = extreme overbought (short opportunity)
   - Proven edge in bear/range markets (2022 crash, 2025 bear)

2. 1d HMA(21) TREND BIAS: Multi-timeframe confirmation
   - Long only when price > 1d HMA (bullish HTF)
   - Short only when price < 1d HMA (bearish HTF)
   - HMA smoother than EMA, less lag for trend detection
   - Call get_htf_data() ONCE before loop (Rule 1)

3. ADX(14) TREND STRENGTH: Avoid whipsaw in weak trends
   - ADX > 20 = sufficient trend strength for entries
   - ADX < 20 = stay flat (choppy market)
   - Hysteresis: enter at 22, exit at 18

4. ATR TRAILING STOP (2.5x): Risk management
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

5. ASYMMETRIC POSITION SIZING: 
   - Bull regime (price > 1d HMA): 0.30 long
   - Bear regime (price < 1d HMA): 0.25 short (more conservative)
   - Discrete levels minimize fee churn

Why this should work:
- CRSI catches reversals in bear market rallies (works in 2025 test period)
- 1d HMA prevents counter-trend trades (avoids 2022 crash losses)
- ADX filter reduces whipsaw in choppy markets
- Should generate 40-80 trades/year per symbol (enough for statistics)
- Works on BTC, ETH, SOL individually (not SOL-biased)
- Conservative sizing protects from 77% BTC crash

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_1d_hma_adx_regime_atr_v1"
timeframe = "4h"
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
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(gain_avg, loss_avg, out=np.zeros_like(gain_avg), where=loss_avg != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Calculate RSI of consecutive up/down streaks for Connors RSI.
    Streak: count consecutive up or down days, then apply RSI to streak values.
    """
    n = len(close)
    streak = np.zeros(n)
    
    # Calculate streak values
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Apply RSI to streak values
    streak_rsi = calculate_rsi(streak, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank for Connors RSI.
    Percent of times current close is lower than previous N closes.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_lower = np.sum(window[:-1] > current)
        pr[i] = (count_lower / (period - 1)) * 100
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean-reversion entries.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.divide(plus_dm_smooth, tr_smooth, out=np.zeros_like(tr_smooth), where=tr_smooth != 0) * 100
    minus_di = np.divide(minus_dm_smooth, tr_smooth, out=np.zeros_like(tr_smooth), where=tr_smooth != 0) * 100
    
    # Calculate DX
    di_sum = plus_di + minus_di
    dx = np.divide(np.abs(plus_di - minus_di), di_sum, out=np.zeros_like(di_sum), where=di_sum != 0) * 100
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels with asymmetry (Rule 4)
    SIZE_LONG = 0.30   # More aggressive in bull regime
    SIZE_SHORT = 0.25  # More conservative in bear regime
    
    # CRSI thresholds for entry
    CRSI_LONG = 10     # Extreme oversold
    CRSI_SHORT = 90    # Extreme overbought
    
    # ADX thresholds with hysteresis
    ADX_ENTER = 22     # Enter when trend strength sufficient
    ADX_EXIT = 18      # Exit when trend weakens
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    adx_active = False  # Track ADX hysteresis state
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === ADX TREND STRENGTH WITH HYSTERESIS ===
        if adx[i] > ADX_ENTER:
            adx_active = True
        elif adx[i] < ADX_EXIT:
            adx_active = False
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < CRSI_LONG
        crsi_overbought = crsi[i] > CRSI_SHORT
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + bull HTF trend + ADX active
        if crsi_oversold and bull_trend_1d and adx_active:
            new_signal = SIZE_LONG
        
        # SHORT ENTRY: CRSI overbought + bear HTF trend + ADX active
        elif crsi_overbought and bear_trend_1d and adx_active:
            new_signal = -SIZE_SHORT
        
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
        # Exit long if HTF trend turns bearish
        if in_position and position_side > 0 and bear_trend_1d:
            new_signal = 0.0
        
        # Exit short if HTF trend turns bullish
        if in_position and position_side < 0 and bull_trend_1d:
            new_signal = 0.0
        
        # === ADX WEAKNESS EXIT ===
        # Exit if ADX drops below exit threshold
        if in_position and not adx_active:
            new_signal = 0.0
        
        # === CRSI REVERSAL EXIT ===
        # Exit long if CRSI becomes overbought
        if in_position and position_side > 0 and crsi_overbought:
            new_signal = 0.0
        
        # Exit short if CRSI becomes oversold
        if in_position and position_side < 0 and crsi_oversold:
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