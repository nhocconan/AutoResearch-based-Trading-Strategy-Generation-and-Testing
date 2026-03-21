#!/usr/bin/env python3
"""
EXPERIMENT #020 - KAMA Trend + RSI Pullback + 4h HMA Filter + BB Regime (30m primary)
=====================================================================================
Hypothesis: 30m KAMA (Kaufman Adaptive Moving Average) adapts to crypto volatility better
than static EMAs. During high volatility, KAMA follows price closely; during chop, it flattens.
Combined with 4h HMA(21) trend filter and Bollinger Band regime detection, we only trade
when volatility expands (BB width > 20th percentile) indicating real moves, not chop.
RSI(14) pullback entries (RSI 35-45 long, 55-65 short) provide better entry timing than
breakouts. This should generate MORE trades than strict Supertrend strategies while
maintaining controlled drawdown via 2*ATR stoploss.

Key features:
- Primary TF: 30m (as required by experiment #020)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(10) adaptive moving average
- Entry: RSI pullback (35-45 long, 55-65 short) - LESS STRICT than before
- Regime: Bollinger Band width > 20th percentile (avoid chop)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work better:
- KAMA adapts to volatility (better than fixed EMA/HMA in crypto)
- BB regime filter avoids trading during low-volatility chop
- RSI range 35-65 is LESS STRICT than 45/55 thresholds (more trades)
- 30m captures more opportunities than 1h/4h strategies
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_4hhma_bbregime_30m_v1"
timeframe = "30m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise/volatility using Efficiency Ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(period, n):
        change[i] = abs(close[i] - close[i - period])
        vol_sum = 0.0
        for j in range(1, period + 1):
            vol_sum += abs(close[i - j + 1] - close[i - j])
        volatility[i] = vol_sum if vol_sum > 0 else 0.0001
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 0.0001, volatility)
    er = change / volatility
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    # Use EMA for smoothing (Wilder's method)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma if sma > 0 else 0
    
    return upper, lower, bandwidth, sma


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    zscore = (close - sma) / std
    zscore = np.where(std == 0, 0, zscore)
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    zscore = calculate_zscore(close, period=20)
    
    # Calculate BB bandwidth percentile for regime filter (rolling 100 bars)
    bb_percentile = np.zeros(n)
    bb_percentile[:] = np.nan
    lookback = 100
    for i in range(lookback, n):
        bb_percentile[i] = np.percentile(bb_bandwidth[i-lookback:i], 20)  # 20th percentile
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.27  # Base position size (27% of capital)
    MAX_SIZE = 0.32   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(bb_bandwidth[i]) or
            np.isnan(zscore[i]) or np.isnan(bb_percentile[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction (slope)
        kama_slope = kama[i] - kama[i - 5] if i >= 5 else 0
        kama_trend = 1 if kama_slope > 0 else -1
        
        # Bollinger Band regime filter (avoid low volatility chop)
        bb_regime_ok = bb_bandwidth[i] > bb_percentile[i]
        
        # RSI pullback conditions - LESS STRICT for more trades
        rsi_pullback_long = 35 <= rsi[i] <= 50  # Pullback in uptrend
        rsi_pullback_short = 50 <= rsi[i] <= 65  # Pullback in downtrend
        
        # Z-score filter (avoid extreme overbought/oversold for entry)
        zscore_ok_long = zscore[i] < 1.5  # Not extremely overbought
        zscore_ok_short = zscore[i] > -1.5  # Not extremely oversold
        
        # Calculate position size (dynamic based on volatility)
        atr_pct = atr[i] / close[i] * 100
        vol_adjustment = min(1.0, 0.03 / atr_pct) if atr_pct > 0 else 1.0
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_adjustment))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + 4h HMA bullish + BB regime ok + RSI pullback + Z-score ok
        if (kama_trend == 1 and hma_trend == 1 and bb_regime_ok and 
            rsi_pullback_long and zscore_ok_long):
            target_signal = position_size
        
        # Short entry: KAMA bearish + 4h HMA bearish + BB regime ok + RSI pullback + Z-score ok
        elif (kama_trend == -1 and hma_trend == -1 and bb_regime_ok and 
              rsi_pullback_short and zscore_ok_short):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * np.sign(position_side)
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if KAMA reverses OR 4h HMA alignment breaks
                kama_reversal_long = kama_trend == -1
                kama_reversal_short = kama_trend == 1
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals