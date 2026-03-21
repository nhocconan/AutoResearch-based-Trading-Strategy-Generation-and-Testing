#!/usr/bin/env python3
"""
EXPERIMENT #005 - KAMA Adaptive Trend + RSI Pullback Strategy (12h)
===================================================================
Hypothesis: 12h timeframe captures multi-day swings while filtering intraday noise.
KAMA adapts to market volatility (fast in trends, slow in ranges), combined with
RSI pullback entries in direction of 1d HMA trend. This should reduce whipsaws
compared to fixed EMAs while capturing sustained moves.

Key features:
- 12h primary timeframe (this experiment's rotation)
- 1d HMA for major trend filter (higher TF reliability)
- KAMA(14) adaptive trend following (responds to volatility changes)
- RSI(14) pullback entries (buy dips in uptrend, sell rallies in downtrend)
- Volume spike confirmation (avoid low-liquidity false signals)
- ATR(14) stoploss at 2.0*ATR with trailing
- Conservative position size (0.25 entry, 0.125 half) for drawdown control
- Fixed discrete signal levels to minimize fee churn

Primary TF: 12h | HTF: 1d HMA trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_12h_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close: np.ndarray, er_period: int = 10, fast_period: int = 2, slow_period: int = 30) -> np.ndarray:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - fast during trends, slow during ranges
    ER (Efficiency Ratio) = |close - close_n| / sum(|close_i - close_i-1|)
    SC (Smoothing Constant) = ER * (fast_sc - slow_sc) + slow_sc
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    # Efficiency Ratio numerator: absolute price change over er_period
    signal = close_s.diff(er_period).abs()
    
    # Efficiency Ratio denominator: sum of absolute single-period changes
    noise = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    # Avoid division by zero
    er = signal / noise.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Dynamic smoothing constant
    sc = (er.values ** 2) * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA iteratively
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = average_gain / average_loss over period
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_values = rsi.values.copy()
    rsi_values[:period + 1] = np.nan
    return rsi_values


def calculate_hma(close: np.ndarray, period: int = 21) -> np.ndarray:
    """
    Calculate Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    More responsive than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half_period, min_periods=half_period, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hull_input = 2 * wma_half - wma_full
    hma = hull_input.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    hma_values = hma.values.copy()
    hma_values[:period] = np.nan
    return hma_values


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values.copy()
    atr[:period] = np.nan
    return atr


def calculate_volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate volume SMA for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values.copy()
    vol_sma[:period] = np.nan
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Extract price data - use .copy() to avoid read-only issues
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values.copy(), period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)  # auto shift(1)
    
    # === CALCULATE 12h INDICATORS (vectorized before loop) ===
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    volume_sma = calculate_volume_sma(volume, period=20)
    
    # Also calculate 12h HMA for additional trend confirmation
    hma_12h = calculate_hma(close, period=21)
    
    # === SIGNAL PARAMETERS ===
    SIZE_ENTRY = 0.25      # 25% position on entry
    SIZE_HALF = 0.125      # 12.5% after take profit
    STOPLOSS_MULT = 2.0    # 2.0*ATR stoploss
    TAKEPROFIT_MULT = 2.0  # 2R take profit
    RSI_OVERSOLD = 35      # RSI < 35 for long entry (pullback)
    RSI_OVERBOUGHT = 65    # RSI > 65 for short entry (rally)
    VOLUME_MULT = 1.1      # Volume must be > 1.1x average for confirmation
    
    signals = np.zeros(n)
    
    # Track position state for stoploss/takeprofit
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    trailing_stop = 0.0
    
    min_lookback = 100  # Ensure all indicators are valid
    
    for i in range(min_lookback, n):
        # Skip if any indicator is NaN
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_sma[i]) or np.isnan(hma_12h[i])):
            signals[i] = 0.0
            continue
        
        current_atr = atr[i]
        current_price = close[i]
        current_rsi = rsi[i]
        current_hma_1d = hma_1d_aligned[i]
        current_hma_12h = hma_12h[i]
        current_kama = kama[i]
        current_volume = volume[i]
        current_volume_sma = volume_sma[i]
        
        # === TREND FILTER (1d HMA) ===
        # Price above 1d HMA = major uptrend, below = major downtrend
        major_trend_up = current_price > current_hma_1d
        major_trend_down = current_price < current_hma_1d
        
        # === 12h TREND CONFIRMATION (KAMA slope) ===
        # KAMA above HMA = bullish, below = bearish
        kama_trend_up = current_kama > current_hma_12h
        kama_trend_down = current_kama < current_hma_12h
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = current_volume > (current_volume_sma * VOLUME_MULT)
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        if position_side == 0:
            # === LONG ENTRY: major uptrend + KAMA uptrend + RSI pullback + volume ===
            if (major_trend_up and kama_trend_up and 
                current_rsi < RSI_OVERSOLD and volume_confirmed):
                new_signal = SIZE_ENTRY
                entry_price = current_price
                position_side = 1
                highest_since_entry = current_price
                lowest_since_entry = current_price
                trailing_stop = entry_price - STOPLOSS_MULT * current_atr
            
            # === SHORT ENTRY: major downtrend + KAMA downtrend + RSI rally + volume ===
            elif (major_trend_down and kama_trend_down and 
                  current_rsi > RSI_OVERBOUGHT and volume_confirmed):
                new_signal = -SIZE_ENTRY
                entry_price = current_price
                position_side = -1
                highest_since_entry = current_price
                lowest_since_entry = current_price
                trailing_stop = entry_price + STOPLOSS_MULT * current_atr
        
        elif position_side == 1:
            # Track highest price since entry for trailing
            highest_since_entry = max(highest_since_entry, current_price)
            
            # === TRAILING STOP: move stop up as price rises ===
            new_trailing_stop = highest_since_entry - STOPLOSS_MULT * current_atr
            if new_trailing_stop > trailing_stop:
                trailing_stop = new_trailing_stop
            
            # === STOPLOSS: price drops below trailing stop ===
            if current_price < trailing_stop:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (current_price - entry_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price + 0.5 * current_atr
            
            # === EXIT: Major trend reversal (1d HMA) ===
            elif major_trend_down:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: KAMA trend reversal ===
            elif kama_trend_down:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: RSI overbought in long position ===
            elif current_rsi > 75:
                new_signal = SIZE_HALF
            
            else:
                # Maintain position
                new_signal = SIZE_ENTRY if new_signal == 0 else new_signal
        
        elif position_side == -1:
            # Track lowest price since entry for trailing
            lowest_since_entry = min(lowest_since_entry, current_price)
            
            # === TRAILING STOP: move stop down as price falls ===
            if trailing_stop == 0:
                trailing_stop = lowest_since_entry + STOPLOSS_MULT * current_atr
            else:
                new_trailing_stop = lowest_since_entry + STOPLOSS_MULT * current_atr
                if new_trailing_stop < trailing_stop:
                    trailing_stop = new_trailing_stop
            
            # === STOPLOSS: price rises above trailing stop ===
            if current_price > trailing_stop:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (entry_price - current_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = -SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price - 0.5 * current_atr
            
            # === EXIT: Major trend reversal (1d HMA) ===
            elif major_trend_up:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: KAMA trend reversal ===
            elif kama_trend_up:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: RSI oversold in short position ===
            elif current_rsi < 25:
                new_signal = -SIZE_HALF
            
            else:
                # Maintain position
                new_signal = -SIZE_ENTRY if new_signal == 0 else new_signal
        
        signals[i] = new_signal
    
    return signals