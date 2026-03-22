#!/usr/bin/env python3
"""
Experiment #422: 30m Volatility Squeeze + 4h HMA Trend + RSI Volume Confirmation

Hypothesis: After 421 experiments, 30m timeframe needs a DIFFERENT approach than 4h/12h.
30m has more noise but also more opportunities. Key insights:

1. VOLATILITY SQUEEZE REGIME (not ADX):
   - Bollinger Band Width at 20-bar low = compression = breakout imminent
   - BB Width percentile < 20% = squeeze (prepare for breakout)
   - BB Width percentile > 60% = expansion (trend following)
   - This works better than ADX on 30m (ADX too laggy for intraday)

2. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - 4h is the RIGHT HTF for 30m (not 1d, which is too slow)
   - 4h HMA smoother than EMA, critical for 30m/4h alignment
   - Long only when price > 4h HMA in squeeze breakout
   - Short only when price < 4h HMA in squeeze breakout

3. RSI(7) FAST MOMENTUM for 30m:
   - RSI(7) instead of RSI(14) for faster 30m signals
   - Long: RSI crosses above 40 from below (momentum building)
   - Short: RSI crosses below 60 from above (momentum fading)
   - More responsive than RSI(14) on 30m timeframe

4. VOLUME CONFIRMATION:
   - Volume > 1.5 * Volume_SMA(20) confirms breakout validity
   - Reduces false breakouts on low-volume 30m bars
   - Critical for 30m where noise is higher than 4h/12h

5. DONCHIAN(20) BREAKOUT TRIGGER:
   - Long: price breaks 20-bar high + volume confirmation + 4h HMA bullish
   - Short: price breaks 20-bar low + volume confirmation + 4h HMA bearish
   - Only enter during BB squeeze (percentile < 20%)

6. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes on 30m timeframe

7. POSITION SIZING: 0.25 discrete (conservative for 30m volatility)
   - Max 25% capital per position (30m more volatile than 4h)
   - Discrete levels minimize fee churn

Why 30m with this approach should work:
- BB squeeze catches breakouts before they happen (leading indicator)
- 4h HMA provides trend filter without being too slow (like 1d)
- RSI(7) + Volume reduces false signals on noisy 30m bars
- Should generate 30-60 trades/year (more than 12h, less than 15m)
- Works on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_squeeze_4h_hma_rsi_vol_donchian_atr_v1"
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
    """Calculate Relative Strength Index with faster period for 30m."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of BB Width over lookback period."""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bandwidth[i-lookback:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid < bandwidth[i]) / len(valid)
            percentile[i] = rank * 100
    
    return percentile

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_width_percentile(bb_bandwidth, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    volume_sma = calculate_volume_sma(volume, 20)
    
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
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === VOLATILITY REGIME DETECTION ===
        bb_squeeze = bb_percentile[i] < 20  # Bandwidth at 20% low = squeeze
        bb_expansion = bb_percentile[i] > 60  # Bandwidth expanding = trend
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_spike = volume[i] > 1.5 * volume_sma[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI MOMENTUM CROSSOVER ===
        rsi_momentum_long = rsi[i] > 40 and rsi[i-1] <= 40  # Cross above 40
        rsi_momentum_short = rsi[i] < 60 and rsi[i-1] >= 60  # Cross below 60
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # BB SQUEEZE BREAKOUT (primary entry signal)
        if bb_squeeze:
            # Long: squeeze + bullish 4h trend + Donchian breakout + volume
            if bull_trend_4h and donchian_long and volume_spike:
                new_signal = SIZE
            # Short: squeeze + bearish 4h trend + Donchian breakout + volume
            elif bear_trend_4h and donchian_short and volume_spike:
                new_signal = -SIZE
        
        # BB EXPANSION TREND FOLLOWING (secondary entry)
        elif bb_expansion:
            # Long: expansion + bullish 4h + RSI momentum
            if bull_trend_4h and rsi_momentum_long:
                new_signal = SIZE
            # Short: expansion + bearish 4h + RSI momentum
            elif bear_trend_4h and rsi_momentum_short:
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
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === RSI EXTREME EXIT (overbought/oversold) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 80:
                new_signal = 0.0  # Take profit on overbought
            if position_side < 0 and rsi[i] < 20:
                new_signal = 0.0  # Take profit on oversold
        
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