#!/usr/bin/env python3
"""
Experiment #368: 30m Multi-Factor Trend Pullback with 4h HMA Bias + Volatility Compression

Hypothesis: After analyzing 367 failed experiments, the pattern is clear:
1. Pure trend-following fails on BTC/ETH (whipsaw in 2022 crash)
2. Pure mean-reversion fails in strong trends
3. Complex regime-switching overfits and fails (-10+ Sharpe in #363, #364)

SOLUTION: Simple multi-factor confirmation with asymmetric logic:
- 4h HMA(21) for stable trend bias (proven in best strategies)
- 30m RSI(7) for pullback entries (faster than RSI(14), catches dips)
- Bollinger Band Width compression for volatility expansion setup
- ADX(14) > 18 to filter choppy markets (looser than 25 to get trades)
- ATR(14) trailing stop at 2.5x for capital protection

Key innovations:
1. ASYMMETRIC ENTRY: Long only when 4h HMA bullish + RSI<35 (pullback in uptrend)
                      Short only when 4h HMA bearish + RSI>65 (rally in downtrend)
2. VOLATILITY FILTER: BB Width must be below 30-day percentile (compression before expansion)
3. CONSERVATIVE SIZING: 0.25 discrete (learned from 77% BTC crash in 2022)
4. FEWER TRADES: Multiple filters = 20-40 trades/year, not 200+ (fee drag killer)

Why 30m works:
- Faster than 4h/12h (more signals) but slower than 5m/15m (less noise)
- 4h HMA provides stable bias without excessive lag
- RSI(7) catches pullbacks that RSI(14) misses
- Should generate 30-50 trades/year per symbol (enough for stats, not too many for fees)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_hma_bb_vol_adx_atr_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 10:
        return adx
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bandwidth = (upper - lower) / sma
    return upper.values, lower.values, bandwidth.values

def calculate_bb_width_percentile(bandwidth, lookback=30):
    """Calculate rolling percentile of BB Width (volatility compression)."""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bandwidth[i-lookback:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bandwidth[i]) / len(valid) * 100
    
    return percentile

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === ADX TREND STRENGTH ===
        # Loosened to 18 to generate more trades (25 was too strict)
        trending_market = adx[i] > 18
        
        # === VOLATILITY COMPRESSION ===
        # BB Width below 40th percentile = compression (setup for expansion)
        vol_compression = bb_width_pct[i] < 40
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI(7) < 35 (oversold pullback in uptrend)
        rsi_oversold = rsi[i] < 35
        
        # Short: RSI(7) > 65 (overbought rally in downtrend)
        rsi_overbought = rsi[i] > 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI pullback + ADX confirms + vol compression
        if bull_trend_4h and rsi_oversold and trending_market and vol_compression:
            new_signal = SIZE
        
        # SHORT ENTRY: 4h bearish + RSI rally + ADX confirms + vol compression
        elif bear_trend_4h and rsi_overbought and trending_market and vol_compression:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === ADX DROPS BELOW THRESHOLD ===
        # Exit if market becomes ranging (ADX < 16 with hysteresis)
        if in_position and adx[i] < 16:
            new_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        # Exit long if RSI goes extremely overbought (>80)
        # Exit short if RSI goes extremely oversold (<20)
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 80:
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 20:
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