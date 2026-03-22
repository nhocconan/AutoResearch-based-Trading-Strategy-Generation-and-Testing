#!/usr/bin/env python3
"""
Experiment #181: 15m Connors RSI Mean Reversion + 4h HMA Trend Filter + BB Regime + ATR Stop

Hypothesis: 15m timeframe is too noisy for trend-following (see exp #169, #175 failures)
but excellent for short-term mean reversion. Connors RSI (CRSI) combines 3 components:
RSI(3) for short-term momentum, RSI-Streak(2) for consecutive up/down days, and
PercentRank(100) for relative position in recent range. CRSI < 10 = extreme oversold,
CRSI > 90 = extreme overbought. This has 75% win rate in literature.

Why this might work on 15m:
- CRSI catches short-term exhaustion that 15m sees well
- 4h HMA provides higher-timeframe bias (only long in 4h uptrend, short in downtrend)
- Bollinger Band width filters regime (wide BB = trending, narrow BB = mean revert)
- ATR stoploss protects against trend continuation against position
- Conservative sizing (0.25) controls drawdown

Key differences from failed 15m strategies:
- #169 used vol_spike + trend following = whipsawed
- #175 used trend_pullback = too many false signals
- This uses MEAN REVERSION which works better on lower timeframes
- CRSI is more robust than simple RSI(14)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_bb_regime_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Short-term momentum
    RSI-Streak(2): Consecutive up/down bar strength
    PercentRank(100): Where price sits in recent range
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    # Streak: consecutive up (+1) or down (-1) bars
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values (treat as price series)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank - where close sits in last N bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i])  # count how many bars below current
        percent_rank[i] = rank / (rank_period - 1) * 100
    
    # Fill initial values
    percent_rank[:rank_period] = 50
    
    # Combine all three components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values, std.values

def calculate_bollinger_bandwidth(upper, lower, sma):
    """Calculate Bollinger Band Width (regime filter)."""
    bw = (upper - lower) / sma
    return bw

def calculate_percentile_bollinger_width(bw, lookback=100):
    """Calculate where current BW sits in recent history (percentile)."""
    n = len(bw)
    bw_percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = bw[i-lookback+1:i+1]
        # Percentile rank of current BW
        bw_percentile[i] = np.sum(window[:-1] <= bw[i]) / (lookback - 1) * 100
    
    bw_percentile[:lookback] = 50
    return bw_percentile

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_sma, bb_std = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bollinger_bandwidth(bb_upper, bb_lower, bb_sma)
    bb_width_pct = calculate_percentile_bollinger_width(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
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
        
        if np.isnan(crsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER ===
        # BB Width percentile < 30 = narrow bands = mean reversion regime
        # BB Width percentile > 70 = wide bands = trending regime (avoid MR)
        mean_reversion_regime = bb_width_pct[i] < 50
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 10 = extreme oversold (long signal)
        # CRSI > 90 = extreme overbought (short signal)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === BOLLINGER BAND POSITION ===
        # Price below lower band = oversold
        # Price above upper band = overbought
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + mean reversion regime + CRSI oversold + price at/near BB lower
        if bull_trend_4h and mean_reversion_regime and crsi_oversold:
            # Need price near or below BB lower for confirmation
            if price_below_bb or close[i] < bb_sma[i] * 0.995:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + mean reversion regime + CRSI overbought + price at/near BB upper
        if bear_trend_4h and mean_reversion_regime and crsi_overbought:
            # Need price near or above BB upper for confirmation
            if price_above_bb or close[i] > bb_sma[i] * 1.005:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === EXIT ON CRSI REVERSAL ===
        # If long and CRSI crosses above 70, take profit
        # If short and CRSI crosses below 30, take profit
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 70:
                new_signal = 0.0  # Take profit on long
            if position_side < 0 and crsi[i] < 30:
                new_signal = 0.0  # Take profit on short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals