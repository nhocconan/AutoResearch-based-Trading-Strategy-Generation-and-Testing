# Strategy: mtf_12h_crsi_choppiness_1d_hma_regime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.122 | +16.0% | -7.8% | 113 | FAIL |
| ETHUSDT | -0.059 | +16.4% | -12.9% | 112 | FAIL |
| SOLUSDT | 0.633 | +77.0% | -18.8% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.163 | +7.8% | -9.3% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #346: 12h Primary + 1d HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: Previous 12h strategies failed because:
1. Too many conflicting filters (regime + trend + momentum all must agree)
2. Symmetric long/short logic whipsawed in 2022 crash and 2025 bear
3. CRSI alone doesn't work without proper regime detection

This strategy uses:
1. 1d HMA(21) as MACRO BIAS (hard filter: only long if 1d bullish, only short if 1d bearish)
2. 12h Choppiness Index for regime detection (CHOP>58=range, CHOP<42=trend)
3. Connors RSI (CRSI) for mean reversion entries in range regime
4. 12h Donchian breakout for trend entries in trend regime
5. ATR-based trailing stop (2.5x ATR) + volume confirmation

KEY INSIGHT: Connors RSI has 75% win rate in range markets. Combined with 1d bias filter,
this reduces whipsaw while capturing mean reversion opportunities. In trend regime,
switch to Donchian breakouts aligned with 1d direction.

TARGET: 25-45 trades/year on 12h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_choppiness_1d_hma_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Standard RSI on 3-period returns
    RSI(streak, 2): RSI on streak duration (consecutive up/down days)
    PercentRank(100): Percentile rank of current return vs last 100 returns
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI on streak duration
    # Streak: count consecutive up/down bars
    returns = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    
    # Calculate RSI on streak (using streak direction as gain/loss)
    streak_delta = pd.Series(streak).diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Component 3: PercentRank of current return vs last 100 returns
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window_returns = returns.iloc[i-pr_period:i].values
        current_return = returns.iloc[i]
        # Count how many returns in window are <= current return
        rank = np.sum(window_returns <= current_return)
        percent_rank[i] = rank / pr_period * 100.0
    
    # CRSI = average of three components
    with np.errstate(invalid='ignore'):
        crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Donchian channels for breakout detection
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 1d HMA for macro bias (HARD FILTER)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (target 25-45 trades/year)
    
    # Position tracking for stoploss/takeprofit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        # Only take LONGS if price above 1d HMA (bullish macro)
        # Only take SHORTS if price below 1d HMA (bearish macro)
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 58.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] < 42.0  # Low choppiness = trend regime (breakout)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Connors RSI mean reversion
            # Long: CRSI < 15 + price above 1d HMA (bullish macro)
            # Short: CRSI > 85 + price below 1d HMA (bearish macro)
            
            if price_above_hma_1d and crsi[i] < 15:
                # Long oversold in bullish macro (range regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and crsi[i] > 85:
                # Short overbought in bearish macro (range regime)
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: Donchian breakout aligned with 1d bias
            # Long: price breaks Donchian upper + 1d bullish
            # Short: price breaks Donchian lower + 1d bearish
            
            breakout_upper = close[i] > donchian_upper[i-1]  # Break above previous upper
            breakout_lower = close[i] < donchian_lower[i-1]  # Break below previous lower
            
            if price_above_hma_1d and breakout_upper and volume_confirmed:
                # Long breakout in bullish macro (trend regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and breakout_lower and volume_confirmed:
                # Short breakdown in bearish macro (trend regime)
                desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL REGIME (42 <= CHOP <= 58): Reduced position size, wait for clarity
            # Only take high-conviction CRSI extremes
            
            if price_above_hma_1d and crsi[i] < 10:
                desired_signal = BASE_SIZE * 0.6
            
            elif price_below_hma_1d and crsi[i] > 90:
                desired_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (3R target) ===
        take_profit_triggered = False
        
        if in_position and position_side > 0:
            profit = close[i] - entry_price
            if profit >= 3.0 * entry_atr:
                take_profit_triggered = True
        
        if in_position and position_side < 0:
            profit = entry_price - close[i]
            if profit >= 3.0 * entry_atr:
                take_profit_triggered = True
        
        if take_profit_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 70:
            # Long position: exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30:
            # Short position: exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered and not take_profit_triggered:
            # Check if regime and bias still valid
            if position_side > 0:
                if (is_choppy and price_above_hma_1d and crsi[i] < 70) or \
                   (is_trending and price_above_hma_1d):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (is_choppy and price_below_hma_1d and crsi[i] > 30) or \
                   (is_trending and price_below_hma_1d):
                    desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 08:56
