# Strategy: mtf_12h_vol_regime_daily_hma_connors_rsi_adx_hysteresis_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.963 | +0.1% | -10.6% | 170 | FAIL |
| ETHUSDT | -0.759 | +2.3% | -12.6% | 172 | FAIL |
| SOLUSDT | 0.061 | +22.8% | -14.9% | 159 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.081 | +6.7% | -5.3% | 62 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #515: 12h Volatility-Regime Adaptive with Daily HMA Bias

Hypothesis: After analyzing 500+ failed experiments, the key insight is that 12h 
timeframe needs VOLATILITY-BASED regime detection rather than just price-based. 
High vol = mean-reversion works (panic reversals). Low vol = trend-following works 
(breakouts have follow-through). Combined with 1d HMA for directional bias.

Key innovations:
1. VOLATILITY REGIME: ATR(7)/ATR(21) ratio > 1.5 = high vol (mean-revert), < 1.0 = low vol (trend)
2. DAILY HMA BIAS: 1d HMA(21) via mtf_data helper for trend direction
3. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for mean-reversion
4. ADX HYSTERESIS: Enter > 22, exit < 18 (prevents whipsaw in borderline cases)
5. LOOSE THRESHOLDS: CRSI < 35 long, > 65 short (ensures ≥10 trades/year)
6. 2.0 * ATR STOPLOSS: Tighter than previous 3.0*ATR for 12h timeframe

Why 12h works:
- Captures multi-day swings without 1d's slowness
- 2 bars/day = enough data for statistical significance
- Less noise than 1h/4h, more signals than 1d
- Volatility regime switching adapts to crypto's changing behavior

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (conservative for 12h swings)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_regime_daily_hma_connors_rsi_adx_hysteresis_atr_v1"
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
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period_rsi)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(period_streak, n):
        if streak[i] > 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        elif streak[i] < 0:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    for i in range(period_rank, n):
        window = returns[i-period_rank+1:i+1]
        count_below = np.sum(window[:-1] < returns[i])
        pct_rank = count_below / (period_rank - 1) * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank) / 3
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_adx_hysteresis(adx, enter_thresh=22, exit_thresh=18):
    """
    Calculate ADX with hysteresis to prevent whipsaw.
    Returns: trending state (1=trending, 0=not trending)
    """
    n = len(adx)
    trending = np.zeros(n)
    state = 0
    
    for i in range(len(adx)):
        if np.isnan(adx[i]):
            continue
        if state == 0 and adx[i] > enter_thresh:
            state = 1
        elif state == 1 and adx[i] < exit_thresh:
            state = 0
        trending[i] = state
    
    return trending

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_21 = calculate_atr(high, low, close, 21)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    adx = calculate_adx(high, low, close, 14)
    adx_trending = calculate_adx_hysteresis(adx, 22, 18)
    
    # Volatility regime: ATR(7)/ATR(21) ratio
    vol_ratio = np.full(n, np.nan)
    for i in range(21, n):
        if atr_21[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_21[i]
    
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
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        high_vol = vol_ratio[i] > 1.5  # Mean-reversion regime
        low_vol = vol_ratio[i] < 1.0   # Trend-following regime
        
        # === ADX TRENDING STATE ===
        is_trending = adx_trending[i] == 1
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # HIGH VOLATILITY: Mean-reversion (panic reversals)
        if high_vol:
            # Long: CRSI oversold + bullish daily bias
            if crsi[i] < 35 and bull_bias:
                new_signal = SIZE
            # Short: CRSI overbought + bearish daily bias
            elif crsi[i] > 65 and bear_bias:
                new_signal = -SIZE
        
        # LOW VOLATILITY: Trend-following (breakouts)
        elif low_vol:
            if is_trending:
                # Long: RSI pullback in uptrend
                if rsi_14[i] < 50 and bull_bias:
                    new_signal = SIZE
                # Short: RSI rally in downtrend
                elif rsi_14[i] > 50 and bear_bias:
                    new_signal = -SIZE
        
        # NEUTRAL VOLATILITY: Mixed approach
        else:
            # Use RSI extremes with daily bias
            if rsi_14[i] < 35 and bull_bias:
                new_signal = SIZE
            elif rsi_14[i] > 65 and bear_bias:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === BIAS REVERSAL EXIT ===
        # Exit if daily trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias and vol_ratio[i] < 1.0:
                new_signal = 0.0
            if position_side < 0 and bull_bias and vol_ratio[i] < 1.0:
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
```

## Last Updated
2026-03-22 18:26
